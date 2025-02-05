from transformers import AutoTokenizer
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from pathlib import Path
import json
import shutil
from typing import Optional, Dict, Any, Union
from enum import Enum

from orbital.tokenizer.config import TokenizerType


class PretrainedVocabulary:
    """
    Loads a pretrained vocabulary from HuggingFace and maps it to the same format
    as VocabularyBuilder output.
    """

    def __init__(self, model_name: str):
        """
        Initialize with a HuggingFace model name.

        Args:
            model_name: Name of the model on HuggingFace (e.g., 'bert-base-uncased')
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab = self.tokenizer.get_vocab()
        self.tokenizer_type = self._detect_tokenizer_type()

    def _detect_tokenizer_type(self) -> TokenizerType:
        """Detect the type of tokenizer based on its characteristics"""
        if hasattr(self.tokenizer, "wordpiece_tokenizer"):
            return TokenizerType.WORDPIECE
        elif getattr(self.tokenizer, "is_fast", False) and isinstance(
            self.tokenizer.backend_tokenizer.model, models.BPE
        ):
            return TokenizerType.BPE
        elif hasattr(self.tokenizer, "sp_model"):
            return TokenizerType.UNIGRAM
        else:
            # Default to WordPiece if can't determine
            return TokenizerType.WORDPIECE

    def _get_special_tokens(self) -> Dict[str, str]:
        """Extract special tokens mapping from the tokenizer"""
        special_tokens = {}

        # Common special tokens to check for
        token_attributes = [
            ("pad_token", "[PAD]"),
            ("unk_token", "[UNK]"),
            ("cls_token", "[CLS]"),
            ("sep_token", "[SEP]"),
            ("mask_token", "[MASK]"),
            ("bos_token", "[BOS]"),
            ("eos_token", "[EOS]"),
        ]

        for attr, default in token_attributes:
            token = getattr(self.tokenizer, attr, None)
            if token is not None:
                special_tokens[attr] = token

        return special_tokens

    def save(self, output_dir: Union[str, Path]) -> None:
        """
        Save the vocabulary and configuration in the same format as VocabularyBuilder.

        Args:
            output_dir: Directory to save the vocabulary and configuration
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save vocabulary
        vocab_path = output_dir / "vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        # Save tokenizer configuration
        config = {
            "tokenizer_type": self.tokenizer_type.value,
            "vocab_size": len(self.vocab),
            "model_name": self.model_name,
            "special_tokens": self._get_special_tokens(),
        }

        config_path = output_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        # Save the tokenizer files
        self.tokenizer.save_pretrained(output_dir)

        print(f"Saved vocabulary and configuration to {output_dir}")
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Tokenizer type: {self.tokenizer_type.value}")

    def get_tokenizer(self) -> Tokenizer:
        """
        Get a Tokenizer instance compatible with VocabularyBuilder output.

        Returns:
            A Tokenizer instance
        """
        # Create a new tokenizer with the same configuration
        if self.tokenizer_type == TokenizerType.WORDPIECE:
            tokenizer = Tokenizer(
                models.WordPiece(vocab=self.vocab, unk_token=self.tokenizer.unk_token)
            )
        elif self.tokenizer_type == TokenizerType.BPE:
            tokenizer = Tokenizer(
                models.BPE(
                    vocab=self.vocab,
                    merges=[],  # Would need to extract merges from the model
                    unk_token=self.tokenizer.unk_token,
                )
            )
        elif self.tokenizer_type == TokenizerType.UNIGRAM:
            max_id = max(self.vocab.values())
            vocab_items = [(token, id / max_id) for token, id in self.vocab.items()]
            tokenizer = Tokenizer(models.Unigram(vocab=vocab_items))
        else:
            tokenizer = Tokenizer(
                models.WordLevel(vocab=self.vocab, unk_token=self.tokenizer.unk_token)
            )

        # Add normalizer
        tokenizer.normalizer = normalizers.Sequence(
            [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
        )

        # Add pre-tokenizer based on type
        if self.tokenizer_type == TokenizerType.BPE:
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        elif self.tokenizer_type == TokenizerType.UNIGRAM:
            tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
        else:
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
                [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
            )

        # Add post-processor
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B [SEP]",
            special_tokens=[
                ("[CLS]", 1),
                ("[SEP]", 2),
            ],
        )

        return tokenizer

    def analyze_coverage(self, text: str) -> list:
        """
        Analyze how well the vocabulary covers the given text.

        Args:
            text: Text to analyze

        Returns:
            List of tuples (word, tokens) for words that got split
        """
        encoding = self.tokenizer(text)
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"])

        # Find words that got split
        words = text.split()
        split_words = []

        for word in words:
            word_tokens = self.tokenizer.tokenize(word)
            if len(word_tokens) > 1:
                split_words.append((word, word_tokens))

        print("\nVocabulary Coverage Analysis:")
        print(f"Total words: {len(words)}")
        print(f"Total tokens: {len(tokens)}")
        print(f"Words split into subwords: {len(split_words)}")

        if split_words:
            print("\nExample splits:")
            for word, tokens in split_words[:5]:
                print(f"  {word} -> {tokens}")

        return split_words
