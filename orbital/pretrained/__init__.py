import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from transformers import AutoTokenizer
import yaml
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import logging

from orbital.files.reader import FileReader
from orbital.tokenizer.config.dataclasses import (
    UnifiedTokenizerConfig,
    load_tokenizer_config,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TokenizerFiles:
    tokenizer_json: Dict[str, Any] | None = None
    tokenizer_json_path: Path | None = None

    tokenizer_config: Dict[str, Any] | None = None
    tokenizer_config_path: Path | None = None

    special_tokens_map: Dict[str, Any] | None = None
    special_tokens_map_path: Path | None = None

    added_tokens: Dict[str, Any] | None = None
    added_tokens_path: Path | None = None

    raw_vocab: Dict[str, int] | None = None
    raw_vocab_path: Path | None = None

    vocab_json: List[str] | None = None
    vocab_json_path: Path | None = None

    vocab_txt: List[str] | None = None
    vocab_txt_path: Path | None = None

    merges_txt: List[str] | None = None
    merges_txt_path: Path | None = None


class TokenizerReader:
    def __init__(self, model_name: str, hugging_cache: Path | None = None):
        self.model_name = model_name

        self.cache_dir = (
            os.path.expanduser(str(hugging_cache))
            if hugging_cache and hugging_cache.exists()
            else os.path.expanduser("~/.cache/huggingface/hub")
        )

        self.model_path = self._get_model_path()
        self.snapshots = self._get_snapshots_path()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

    def _get_model_path(self) -> Path:
        if self.cache_dir:
            base = Path(self.cache_dir)
        else:
            base = Path(os.path.expanduser("~/.cache/huggingface/hub"))
        return base / f"models--{self.model_name.replace('/', '--')}"

    def _get_snapshots_path(self) -> Path:
        snapshots = self.model_path / "snapshots"
        if not snapshots.exists():
            raise ValueError(f"No snapshots found for {self.model_name}")
        return next(snapshots.iterdir())  # Get latest snapshot

    def get_tokenizer_files(self) -> TokenizerFiles:
        tokenizer_json, tokenizer_json_path = FileReader._safe_read_json(
            "tokenizer.json", self.snapshots
        )
        tokenizer_config, tokenizer_config_path = FileReader._safe_read_json(
            "tokenizer_config.json", self.snapshots
        )
        special_tokens_map, special_tokens_map_path = FileReader._safe_read_json(
            "special_tokens_map.json", self.snapshots
        )
        added_tokens, added_tokens_path = FileReader._safe_read_json(
            "added_tokens.json", self.snapshots
        )

        vocab_txt, vocab_txt_path = FileReader._safe_read_text(
            "vocab.txt", self.snapshots
        )
        vocab_json, vocab_json_path = FileReader._safe_read_json(
            "vocab.json", self.snapshots
        )

        merges_txt, merges_txt_path = FileReader._safe_read_text(
            "merges.txt", self.snapshots
        )

        files = TokenizerFiles(
            tokenizer_json=tokenizer_json,
            tokenizer_json_path=tokenizer_json_path,
            tokenizer_config=tokenizer_config,
            tokenizer_config_path=tokenizer_config_path,
            special_tokens_map=special_tokens_map,
            special_tokens_map_path=special_tokens_map_path,
            added_tokens=added_tokens,
            added_tokens_path=added_tokens_path,
            vocab_json=vocab_json,
            vocab_json_path=vocab_json_path,
            vocab_txt=vocab_txt,
            vocab_txt_path=vocab_txt_path,
            merges_txt=merges_txt,
            merges_txt_path=merges_txt_path,
            raw_vocab=self.tokenizer.get_vocab(),
        )
        return files

    def _determine_tokenizer_type(self, files: TokenizerFiles) -> str:
        # First check tokenizer_json if available
        if files.tokenizer_json and "model" in files.tokenizer_json:
            model_type = files.tokenizer_json["model"]["type"].lower()
            if "bpe" in model_type:
                return "BPE"
            elif "wordpiece" in model_type:
                return "WordPiece"
            elif "unigram" in model_type:
                return "SentencePiece"

        # Then check tokenizer_config
        if files.tokenizer_config and "tokenizer_class" in files.tokenizer_config:
            class_name = files.tokenizer_config["tokenizer_class"].lower()
            if "bpe" in class_name:
                return "BPE"
            elif "wordpiece" in class_name:
                return "WordPiece"
            elif "unigram" in class_name or "sentencepiece" in class_name:
                return "SentencePiece"

        # Fallback detection based on files
        if files.merges_txt:
            return "BPE"
        elif files.vocab_txt:
            return "WordPiece"
        else:
            return "Unknown"

    def analyze_vocabulary(self) -> Dict[str, Any]:
        files = self.get_tokenizer_files()
        tokenizer_type = self._determine_tokenizer_type(files)

        analysis = {
            "model_name": self.model_name,
            "tokenizer_type": tokenizer_type,
            "vocab_size": len(files.raw_vocab) if files.raw_vocab else None,
            "model_max_length": getattr(self.tokenizer, "model_max_length", None),
            "special_tokens": files.special_tokens_map,
            "added_tokens": files.added_tokens,
            "vocabulary_stats": self._analyze_vocab_stats(files),
        }

        if tokenizer_type == "BPE" and files.merges_txt:
            analysis["merges_sample"] = files.merges_txt[:10]

        return {k: v for k, v in analysis.items() if v is not None}

    def _analyze_vocab_stats(self, files: TokenizerFiles) -> Dict[str, Any]:
        vocab = files.raw_vocab or {}
        return {
            "num_single_chars": sum(1 for t in vocab if len(t) == 1),
            "num_subwords": sum(1 for t in vocab if len(t) > 1),
            "sample_tokens": list(vocab.items())[:10],
            "has_merges": bool(files.merges_txt),
            "has_vocab_txt": bool(files.vocab_txt),
        }


class TokenizerAggregator:
    def __init__(
        self,
        model_names: List[str],
        output_dir: Path,
        hugging_cache: Path | None = None,
    ):
        self.model_names = model_names
        self.output_dir = Path(output_dir)
        self.hugging_cache = hugging_cache
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_model(self, model_name: str) -> Dict[str, Any]:
        reader = TokenizerReader(model_name, self.hugging_cache)
        files = reader.get_tokenizer_files()

        analysis = reader.analyze_vocabulary()

        config = {"tokenizer_files": asdict(files), "analysis": analysis}

        output_path = (
            self.output_dir / f"{model_name.replace('/', '_')}_tokenizer_config.yaml"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)

        return config

    def aggregate_configs(self, max_workers: int = 4) -> Dict[str, Dict[str, Any]]:
        configs = {}
        for model_name in self.model_names:
            configs[model_name] = self.process_model(model_name)

        return configs


def from_pretrained(
    model_name: str, hugging_cache: Optional[Path] = None
) -> UnifiedTokenizerConfig:
    """
    Use TokenizerReader to:
      1) Download/cache the HF model files.
      2) Inspect the tokenizer JSON/config/etc.
      3) Determine the subword algorithm (BPE, WordPiece, etc.)
      4) Build a corresponding UnifiedTokenizerConfig

    :param model_name: A model identifier (e.g., "bert-base-uncased")
                       or local path recognized by AutoTokenizer.
    :param hugging_cache: Optional custom path for HF caching.
    :return: A ModelConfig populated with the discovered parameters.
    """
    reader = TokenizerReader(model_name=model_name, hugging_cache=hugging_cache)
    files = (
        reader.get_tokenizer_files()
    )  # Will contain merges_txt, vocab_txt, raw_vocab, etc.
    analysis = reader.analyze_vocabulary()
    config = {"tokenizer_files": asdict(files), "analysis": analysis}
    model_config = load_tokenizer_config(config)
    return model_config
