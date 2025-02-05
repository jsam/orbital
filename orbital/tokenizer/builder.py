from typing import List, Union
from tokenizers import (
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    decoders,
    processors,
)
from tokenizers.trainers import (
    WordPieceTrainer,
    BpeTrainer,
    UnigramTrainer,
    WordLevelTrainer,
)
from pathlib import Path
from collections import Counter
import regex as re
import json
from tqdm import tqdm
import numpy as np

from orbital.tokenizer.config import TokenizerType


class VocabularyBuilder:
    def __init__(
        self,
        tokenizer_type: TokenizerType,
        vocab_size=30000,
        min_frequency=2,
        special_tokens=None,
    ):
        self.tokenizer_type = tokenizer_type
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

        # Define default special tokens if none provided
        self.special_tokens = special_tokens or [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
        ]

        # Initialize the appropriate tokenizer model based on type
        self.tokenizer = self._create_base_tokenizer()

        # Add normalizer (common for all types)
        self.tokenizer.normalizer = normalizers.Sequence(
            [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
        )

        # Add pre-tokenizer (can be customized per type if needed)
        self.tokenizer.pre_tokenizer = self._get_pre_tokenizer()

        # Add post-processor
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B [SEP]",
            special_tokens=[
                ("[CLS]", 1),
                ("[SEP]", 2),
            ],
        )

        # Initialize counter
        self.word_counter = Counter()

    def _create_base_tokenizer(self) -> Tokenizer:
        """Create the appropriate tokenizer model based on type"""
        if self.tokenizer_type == TokenizerType.WORDPIECE:
            return Tokenizer(
                models.WordPiece(unk_token="[UNK]", max_input_chars_per_word=100)
            )
        elif self.tokenizer_type == TokenizerType.BPE:
            return Tokenizer(models.BPE(unk_token="[UNK]"))
        elif self.tokenizer_type == TokenizerType.UNIGRAM:
            return Tokenizer(models.Unigram())
        elif self.tokenizer_type == TokenizerType.WORDLEVEL:
            return Tokenizer(models.WordLevel(unk_token="[UNK]"))
        else:
            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")

    def _get_pre_tokenizer(self):
        """Get the appropriate pre-tokenizer based on tokenizer type"""
        if self.tokenizer_type == TokenizerType.WORDPIECE:
            return pre_tokenizers.Sequence(
                [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
            )
        elif self.tokenizer_type == TokenizerType.BPE:
            return pre_tokenizers.ByteLevel()
        elif self.tokenizer_type == TokenizerType.UNIGRAM:
            return pre_tokenizers.Metaspace()
        elif self.tokenizer_type == TokenizerType.WORDLEVEL:
            return pre_tokenizers.Whitespace()
        else:
            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")

    def _get_trainer(self):
        """Get the appropriate trainer based on tokenizer type"""
        common_params = {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "special_tokens": self.special_tokens,
        }

        if self.tokenizer_type == TokenizerType.WORDPIECE:
            return WordPieceTrainer(continuing_subword_prefix="##", **common_params)
        elif self.tokenizer_type == TokenizerType.BPE:
            return BpeTrainer(**common_params)
        elif self.tokenizer_type == TokenizerType.UNIGRAM:
            return UnigramTrainer(**common_params)
        elif self.tokenizer_type == TokenizerType.WORDLEVEL:
            return WordLevelTrainer(**common_params)
        else:
            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")

    def analyze_directory(
        self, directory_path: Union[str, Path]
    ) -> "VocabularyBuilder":
        """
        Analyze all text files in a directory and train the tokenizer.

        Args:
            directory_path: Path to directory containing .txt files
        """
        data_dir = Path(directory_path)
        if not data_dir.is_dir():
            raise ValueError(f"Directory not found: {data_dir}")

        files = list(data_dir.glob("**/*.txt"))
        print(f"Found {len(files)} text files in directory")

        texts = []
        for file_path in tqdm(files, desc="Reading files"):
            texts.extend(self._read_file(file_path))

        return self._train_tokenizer(texts)

    def analyze_texts(self, texts: List[str]) -> "VocabularyBuilder":
        """
        Analyze a list of text strings and train the tokenizer.

        Args:
            texts: List of text strings to analyze
        """
        if not texts:
            raise ValueError("Empty text list provided")

        print(f"Processing {len(texts)} text strings")

        # Update word counter for statistics
        for text in tqdm(texts, desc="Analyzing texts"):
            words = re.findall(r"\b\w+\b", text.lower())
            self.word_counter.update(words)

        return self._train_tokenizer(texts)

    def analyze_files(self, file_paths: List[Union[str, Path]]) -> "VocabularyBuilder":
        """
        Analyze a list of text files and train the tokenizer.

        Args:
            file_paths: List of paths to .txt files
        """
        if not file_paths:
            raise ValueError("Empty file list provided")

        print(f"Processing {len(file_paths)} files")

        texts = []
        for file_path in tqdm(file_paths, desc="Reading files"):
            texts.extend(self._read_file(Path(file_path)))

        return self._train_tokenizer(texts)

    def _train_tokenizer(self, texts: List[str]) -> "VocabularyBuilder":
        """
        Train the tokenizer on the collected texts.

        Args:
            texts: List of text strings to train on
        """
        print(f"\nTotal texts collected: {len(texts)}")

        print("\nVocabulary Statistics:")
        print(f"Total unique words: {len(self.word_counter)}")
        print(
            f"Words appearing >= {self.min_frequency} times: "
            f"{sum(1 for count in self.word_counter.values() if count >= self.min_frequency)}"
        )

        # Get the appropriate trainer
        trainer = self._get_trainer()

        # Train the tokenizer
        self.tokenizer.train_from_iterator(texts, trainer=trainer)

        return self

    def _read_file(self, file_path: Path) -> list:
        """Read a text file and return its contents as a list of strings"""
        try:
            text = file_path.read_text(encoding="utf-8")

            # Count words for statistics
            words = re.findall(r"\b\w+\b", text.lower())
            self.word_counter.update(words)

            return [text]
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []

        print("\nVocabulary Statistics:")
        print(f"Total unique words: {len(self.word_counter)}")
        print(
            f"Words appearing >= {self.min_frequency} times: "
            f"{sum(1 for count in self.word_counter.values() if count >= self.min_frequency)}"
        )

        # Get the appropriate trainer
        trainer = self._get_trainer()

        # Train the tokenizer
        self.tokenizer.train_from_iterator(texts, trainer=trainer)

        return self

    def build_tokenizer(self, output_dir="custom_tokenizer"):
        """Save the trained tokenizer"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Save the tokenizer
        self.tokenizer.save(str(output_dir / "tokenizer.json"))

        # Get and save vocabulary
        vocab = self.tokenizer.get_vocab()
        with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

        # Save tokenizer configuration
        config = {
            "tokenizer_type": self.tokenizer_type.value,
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "special_tokens": self.special_tokens,
        }

        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nTokenizer built and saved to {output_dir}")
        print(f"Vocabulary size: {len(vocab)}")
        print(f"Tokenizer type: {self.tokenizer_type.value}")

        return self.tokenizer

    def analyze_coverage(self, text):
        """Analyze how well the vocabulary covers the given text"""
        # Encode the text
        encoding = self.tokenizer.encode(text)
        tokens = encoding.tokens

        # Find words that got split
        words = re.findall(r"\b\w+\b", text)
        split_words = []

        for word in words:
            word_encoding = self.tokenizer.encode(word)
            if len(word_encoding.tokens) > 1:
                split_words.append((word, word_encoding.tokens))

        print("\nVocabulary Coverage Analysis:")
        print(f"Total words: {len(words)}")
        print(f"Total tokens: {len(tokens)}")
        print(f"Words split into subwords: {len(split_words)}")

        if split_words:
            print("\nExample splits:")
            for word, tokens in split_words[:5]:
                print(f"  {word} -> {tokens}")

        return split_words
