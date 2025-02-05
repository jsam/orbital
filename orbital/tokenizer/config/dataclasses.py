from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from enum import Enum

from orbital.files.reader import FileReader

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class VocabContent:
    """
    Container for vocabulary content with type information.

    Attributes:
        content: The vocabulary content, either as a dictionary mapping tokens to IDs
                or as a list of tokens where the index represents the token ID
        source_path: Path to the file from which the vocabulary was loaded
        format_type: Indicates whether the vocabulary is from a JSON or text file
    """

    content: Dict[str, int]
    source_path: Path
    format_type: str  # 'json' or 'text'


@dataclass
class BPEMergeRule:
    """
    Represents a single BPE merge rule.

    Attributes:
        pair: Tuple of tokens to be merged
        priority: Index representing merge priority (lower means higher priority)
    """

    pair: tuple[str, str]
    priority: int

    @classmethod
    def from_line(cls, line: str, priority: int) -> Optional["BPEMergeRule"]:
        """
        Create a BPEMergeRule from a line in the merges.txt file.

        Args:
            line: A line from the merges.txt file (e.g., "t h")
            priority: The line number / priority of this merge rule

        Returns:
            BPEMergeRule if the line is valid, None otherwise
        """
        try:
            tokens = line.strip().split()
            if len(tokens) != 2:
                logger.warning(
                    f"Invalid merge rule format at priority {priority}: {line}"
                )
                return None
            return cls(pair=(tokens[0], tokens[1]), priority=priority)
        except Exception as e:
            logger.warning(
                f"Failed to parse merge rule at priority {priority}: {str(e)}"
            )
            return None


@dataclass
class BPEMerges:
    """
    Container for BPE merge rules with ordering information.

    Attributes:
        rules: List of BPE merge rules in priority order
        source_path: Path to the merges.txt file
    """

    rules: List[BPEMergeRule]
    source_path: Path

    @property
    def num_merges(self) -> int:
        return len(self.rules)

    def get_merge_priority(self, token1: str, token2: str) -> Optional[int]:
        """
        Get the priority of a specific merge pair.

        Args:
            token1: First token in the pair
            token2: Second token in the pair

        Returns:
            Priority of the merge rule if found, None otherwise
        """
        for rule in self.rules:
            if rule.pair == (token1, token2):
                return rule.priority
        return None


@dataclass
class TokenizerFilePaths:
    tokenizer_json_path: Path | None = None
    tokenizer_config_path: Path | None = None
    special_tokens_map_path: Path | None = None
    added_tokens_path: Path | None = None
    raw_vocab_path: Path | None = None
    vocab_json_path: Path | None = None
    vocab_txt_path: Path | None = None
    merges_txt_path: Path | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "TokenizerFilePaths":
        """
        Create a TokenizerFilePaths instance from a dictionary.

        Args:
            data: Dictionary containing path information

        Returns:
            TokenizerFilePaths: Instance with paths converted to Path objects
        """

        def to_path(value: str | None) -> Path | None:
            return Path(value) if value else None

        return cls(
            tokenizer_json_path=to_path(data.get("tokenizer_json_path")),
            tokenizer_config_path=to_path(data.get("tokenizer_config_path")),
            special_tokens_map_path=to_path(data.get("special_tokens_map_path")),
            added_tokens_path=to_path(data.get("added_tokens_path")),
            raw_vocab_path=to_path(data.get("raw_vocab_path")),
            vocab_json_path=to_path(data.get("vocab_json_path")),
            vocab_txt_path=to_path(data.get("vocab_txt_path")),
            merges_txt_path=to_path(data.get("merges_txt_path")),
        )

    def read_vocab(self) -> Optional[VocabContent]:
        """
        Get the vocabulary from either JSON or text file.

        Returns:
            VocabContent containing the vocabulary and its metadata if successful,
            None if no vocabulary could be loaded
        """
        # NOTE: We give vocab.json higher priority if it exists.
        if self.vocab_json_path:
            vocab_json, path = FileReader._safe_read_json(str(self.vocab_json_path))
            if vocab_json is not None and path is not None:
                return VocabContent(
                    content=vocab_json, source_path=path, format_type="json"
                )

        if self.vocab_txt_path:
            vocab_txt, path = FileReader._safe_read_text(str(self.vocab_txt_path))
            if vocab_txt is not None and path is not None:
                return VocabContent(
                    content=vocab_txt, source_path=path, format_type="text"
                )

        logger.warning("No vocabulary file found or could be read")
        return None

    def read_merges(self) -> Optional[BPEMerges]:
        """
        Get the BPE merge rules from the merges.txt file.

        The merges.txt file contains the ordered list of token merges learned during
        BPE training. Each line represents one merge rule showing which pairs should
        be merged. The order is crucial as it determines the sequence of merge
        operations during tokenization.

        Returns:
            BPEMerges containing the ordered merge rules if successful,
            None if no merges file could be loaded
        """
        if not self.merges_txt_path:
            logger.warning("No merges file path specified")
            return None

        merges_txt, path = FileReader._safe_read_text(str(self.merges_txt_path))
        if merges_txt is None or path is None:
            logger.warning("Could not read merges file")
            return None

        # Parse merge rules with their priorities
        rules = []
        for i, line in enumerate(merges_txt):
            rule = BPEMergeRule.from_line(line, priority=i)
            if rule is not None:
                rules.append(rule)

        if not rules:
            logger.warning("No valid merge rules found in file")
            return None

        return BPEMerges(rules=rules, source_path=path)


class TokenizerType(Enum):
    BPE = "bpe"
    UNIGRAM = "unigram"
    WORDLEVEL = "wordlevel"
    WORDPIECE = "wordpiece"

    @classmethod
    def from_str(cls, value: str) -> "TokenizerType":
        """Convert string to TokenizerType, handling case variations"""
        try:
            return cls(value)
        except ValueError:
            # Special case for WordPiece
            if value.lower() == "wordpiece":
                return cls.WORDPIECE
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
            raise ValueError(f"Invalid TokenizerType: {value}")


class PaddingSide(Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclass
class SpecialToken:
    """Represents a special token with its properties"""

    content: str
    id: int
    special: bool = True
    single_word: bool = False
    lstrip: bool = False
    rstrip: bool = False
    normalized: bool = False


@dataclass
class TokenizerComponent:
    """Base class for tokenizer components"""

    type: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> Optional["TokenizerComponent"]:
        if not data:
            return None
        component_type = data.pop("type")
        return cls(type=component_type, parameters=data)


@dataclass
class TokenizerNormalizer:
    """Configuration for text normalization"""

    lowercase: bool = False
    strip_accents: bool = False
    clean_text: bool = True
    handle_chinese_chars: bool = True
    replace_spaces: Optional[str] = None  # For SentencePiece's ▁
    prepend_space: bool = False

    # Support for sequence of normalizers
    normalizers: List[TokenizerComponent] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> Optional["TokenizerNormalizer"]:
        if not data:
            return None

        if data.get("type") == "BertNormalizer":
            return cls(
                lowercase=data.get("lowercase", False),
                strip_accents=data.get("strip_accents", False),
                clean_text=data.get("clean_text", True),
                handle_chinese_chars=data.get("handle_chinese_chars", True),
            )
        elif data.get("type") == "Sequence":
            normalizers = [
                TokenizerComponent.from_dict(n) for n in data.get("normalizers", [])
            ]
            return cls(
                normalizers=normalizers,
                replace_spaces=next(
                    (
                        n.parameters.get("content")
                        for n in normalizers
                        if n.type == "Replace"
                        and n.parameters.get("pattern", {}).get("String") == " "
                    ),
                    None,
                ),
                prepend_space=any(n.type == "Prepend" for n in normalizers),
            )
        return None


@dataclass
class PreTokenizationConfig:
    """Configuration for pre-tokenization steps"""

    add_prefix_space: bool = False
    trim_offsets: bool = True
    split_special_tokens: bool = False
    split_pattern: Optional[str] = None  # Regex pattern for splitting
    byte_fallback: bool = True

    # Support for sequence of pretokenizers
    pretokenizers: List[TokenizerComponent] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> Optional["PreTokenizationConfig"]:
        if not data:
            return None
        pretokenizers = []
        if data.get("type") == "Sequence":
            pretokenizers = [
                TokenizerComponent.from_dict(p) for p in data.get("pretokenizers", [])
            ]
        return cls(
            pretokenizers=pretokenizers,
            split_pattern=next(
                (
                    p.parameters.get("pattern", {}).get("String")
                    for p in pretokenizers
                    if p.type == "Split"
                ),
                None,
            ),
            byte_fallback=any(p.type == "ByteLevel" for p in pretokenizers),
        )


@dataclass
class ModelConfig:
    """Core tokenizer model configuration"""

    type: TokenizerType
    vocab_size: int
    unk_token: str = "<unk>"
    continuing_subword_prefix: Optional[str] = None  # e.g., "##" for WordPiece
    max_input_chars_per_word: int = 100
    dropout: Optional[float] = None
    fuse_unk: bool = False
    byte_fallback: bool = False

    def __post_init__(self):
        if self.vocab_size < 0:
            raise ValueError("vocab_size must be non-negative")

    @classmethod
    def from_dict(
        cls, data: dict, tokenizer_type: str, vocab_size: int
    ) -> "ModelConfig":
        """Create ModelConfig from dictionary and analysis data"""
        return cls(
            type=TokenizerType.from_str(tokenizer_type),
            vocab_size=vocab_size,  # Use vocab_size from analysis
            unk_token=data.get("unk_token"),
            continuing_subword_prefix=data.get("continuing_subword_prefix"),
            max_input_chars_per_word=data.get("max_input_chars_per_word", 100),
            dropout=data.get("dropout"),
            fuse_unk=data.get("fuse_unk", False),
            byte_fallback=data.get("byte_fallback", False),
        )


@dataclass
class PostProcessorConfig:
    """Configuration for post-processing"""

    type: str = "TemplateProcessing"
    single: List[Dict[str, Any]] = field(default_factory=list)
    pair: List[Dict[str, Any]] = field(default_factory=list)
    special_tokens: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> Optional["PostProcessorConfig"]:
        if not data:
            return None
        return cls(
            type=data.get("type", "TemplateProcessing"),
            single=data.get("single", []),
            pair=data.get("pair", []),
            special_tokens=data.get("special_tokens", {}),
        )


@dataclass
class UnifiedTokenizerConfig:
    """Main configuration class for the unified tokenizer"""

    model: ModelConfig
    name: str
    version: str = "1.0"
    model_max_length: int = 2048
    padding_side: PaddingSide = PaddingSide.RIGHT
    special_tokens: Dict[str, SpecialToken] = field(default_factory=dict)

    # Components
    normalizer: Optional[TokenizerNormalizer] = None
    pre_tokenization: Optional[PreTokenizationConfig] = None
    post_processor: Optional[PostProcessorConfig] = None

    # Additional settings
    clean_up_tokenization_spaces: bool = True
    add_prefix_space: bool = False
    legacy: bool = False
    chat_template: Optional[str] = None
    files: TokenizerFilePaths = field(default_factory=TokenizerFilePaths)

    @property
    def vocab_size(self) -> int:
        """Get total vocab size including special tokens"""
        return self.model.vocab_size + len(self.special_tokens)

    @classmethod
    def from_dict(cls, data: dict) -> "UnifiedTokenizerConfig":
        """Create UnifiedTokenizerConfig from dictionary"""
        tokenizer_files = data.get("tokenizer_files", {})
        tokenizer_json = tokenizer_files.get("tokenizer_json", {})
        tokenizer_config = tokenizer_files.get("tokenizer_config", {})
        analysis = data.get("analysis", {})

        # Process file paths
        files = TokenizerFilePaths.from_dict(tokenizer_files)

        # Process special tokens
        special_tokens = {}
        for token in tokenizer_json.get("added_tokens", []):
            special_tokens[token["content"]] = SpecialToken(
                content=token["content"],
                id=token["id"],
                special=token.get("special", True),
                single_word=token.get("single_word", False),
                lstrip=token.get("lstrip", False),
                rstrip=token.get("rstrip", False),
                normalized=token.get("normalized", False),
            )

        return cls(
            model=ModelConfig.from_dict(
                tokenizer_json.get("model", {}),
                analysis.get("tokenizer_type", "bpe"),
                analysis.get("vocab_size", 0),  # Pass vocab_size from analysis
            ),
            name=analysis.get("model_name", ""),
            version=tokenizer_json.get("version", "1.0"),
            model_max_length=tokenizer_config.get("model_max_length", None),
            special_tokens=special_tokens,
            normalizer=TokenizerNormalizer.from_dict(tokenizer_json.get("normalizer")),
            pre_tokenization=PreTokenizationConfig.from_dict(
                tokenizer_json.get("pre_tokenizer")
            ),
            post_processor=PostProcessorConfig.from_dict(
                tokenizer_json.get("post_processor")
            ),
            clean_up_tokenization_spaces=tokenizer_config.get(
                "clean_up_tokenization_spaces", True
            ),
            add_prefix_space=tokenizer_config.get("add_prefix_space", False),
            legacy=tokenizer_config.get("legacy", False),
            chat_template=tokenizer_config.get("chat_template"),
            files=files,
        )


def load_tokenizer_config(config_dict: dict) -> UnifiedTokenizerConfig:
    """
    Load a tokenizer configuration from a dictionary.

    Args:
        config_dict: Dictionary containing the tokenizer configuration

    Returns:
        UnifiedTokenizerConfig: A unified tokenizer configuration object
    """
    return UnifiedTokenizerConfig.from_dict(config_dict)
