import pytest
from orbital.tokenizer.config import TokenizerType
from orbital.pretrained import from_pretrained

MODELS = [
    ("microsoft/phi-2", TokenizerType.BPE, 50295, 2048),
    ("microsoft/Phi-3.5-mini-instruct", TokenizerType.BPE, 32011, 131072),
    ("Qwen/Qwen2.5-Coder-3B-Instruct", TokenizerType.BPE, 151665, 32768),
    (
        "mistralai/Mistral-7B-v0.1",
        TokenizerType.BPE,
        32000,
        1000000000000000019884624838656,  # context size here is broken ... why?
    ),
]


@pytest.mark.parametrize("model_name,expected_type,base_vocab,model_max_length", MODELS)
def test_model_configurations(
    model_name: str,
    expected_type: TokenizerType,
    base_vocab: int,
    model_max_length: int,
):
    """Test loading configurations from the actual models"""
    config = from_pretrained(model_name)

    # Test model type
    assert config.model.type == expected_type
    assert config.model_max_length == model_max_length

    # vocab_size = len(config.files.read_vocab().content)
    # special_token_size = len(config.special_tokens)
    # assert vocab_size + special_token_size == base_vocab

    # Test basic properties are populated
    assert config.model.vocab_size == base_vocab
    assert len(config.special_tokens) > 0
    assert config.model_max_length > 0


import pytest
from pathlib import Path
import shutil
import json
from unittest.mock import Mock, patch, MagicMock
from transformers import PreTrainedTokenizer
from tokenizers import Tokenizer
from tokenizers.models import WordPiece


from orbital.tokenizer import PretrainedVocabulary, TokenizerType


@pytest.fixture
def mock_tokenizer():
    """Create a mock HuggingFace tokenizer"""
    # Create the base mock
    mock = Mock(spec=PreTrainedTokenizer)

    # Basic tokenizer attributes
    mock.get_vocab.return_value = {"test": 0, "word": 1, "[PAD]": 2, "[UNK]": 3}
    mock.unk_token = "[UNK]"
    mock.pad_token = "[PAD]"
    mock.cls_token = "[CLS]"
    mock.sep_token = "[SEP]"
    mock.mask_token = "[MASK]"

    # Tokenization methods
    mock.tokenize.return_value = ["test"]
    mock.convert_ids_to_tokens.return_value = ["test"]

    # Set up the __call__ behavior using side_effect
    def tokenizer_call(*args, **kwargs):
        return {"input_ids": [0, 1, 2]}

    mock.side_effect = tokenizer_call

    # Backend tokenizer setup
    mock_backend = Mock()
    mock_backend.model = Mock(spec=WordPiece)
    mock.backend_tokenizer = mock_backend

    # Fast tokenizer attribute
    mock.is_fast = True

    return mock


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files"""
    test_dir = tmp_path / "test_output"
    yield test_dir
    # Cleanup
    if test_dir.exists():
        shutil.rmtree(test_dir)


class TestPretrainedVocabulary:
    @pytest.fixture(autouse=True)
    def setup(self, mock_tokenizer, monkeypatch):
        """Setup test environment before each test"""

        def mock_from_pretrained(*args, **kwargs):
            return mock_tokenizer

        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained", mock_from_pretrained
        )
        self.vocab = PretrainedVocabulary("bert-base-uncased")
        self.mock_tokenizer = mock_tokenizer

    def test_initialization(self):
        """Test initialization with model name"""
        assert self.vocab.model_name == "bert-base-uncased"
        assert isinstance(self.vocab.vocab, dict)
        assert isinstance(self.vocab.tokenizer_type, TokenizerType)

    @pytest.mark.parametrize(
        "tokenizer_attr,expected_type",
        [
            ("wordpiece_tokenizer", TokenizerType.WORDPIECE),
            ("sp_model", TokenizerType.UNIGRAM),
        ],
    )
    def test_tokenizer_type_detection(self, tokenizer_attr, expected_type):
        """Test detection of different tokenizer types"""
        setattr(self.mock_tokenizer, tokenizer_attr, Mock())
        tokenizer_type = self.vocab._detect_tokenizer_type()
        assert tokenizer_type == expected_type

    def test_special_tokens_extraction(self):
        """Test extraction of special tokens"""
        special_tokens = self.vocab._get_special_tokens()
        assert isinstance(special_tokens, dict)
        assert special_tokens["unk_token"] == "[UNK]"
        assert special_tokens["pad_token"] == "[PAD]"

    def test_save(self, temp_dir):
        """Test saving vocabulary and configuration"""
        with patch.object(self.mock_tokenizer, "save_pretrained"):
            self.vocab.save(temp_dir)

            # Check files were created
            assert (temp_dir / "vocab.json").exists()
            assert (temp_dir / "config.json").exists()

            # Check config contents
            with open(temp_dir / "config.json") as f:
                config = json.load(f)
                assert config["model_name"] == "bert-base-uncased"
                assert config["vocab_size"] == len(self.vocab.vocab)
                assert "special_tokens" in config

    def test_get_tokenizer(self):
        """Test getting a tokenizer instance"""
        tokenizer = self.vocab.get_tokenizer()
        assert isinstance(tokenizer, Tokenizer)
        assert tokenizer.get_vocab() == self.vocab.vocab

    def test_analyze_coverage(self):
        """Test vocabulary coverage analysis"""
        # Mock tokenize method to simulate word splitting
        self.mock_tokenizer.tokenize.side_effect = (
            lambda word: [word[0:2], "##" + word[2:]] if len(word) > 4 else [word]
        )

        text = "testing vocabulary coverage"
        split_words = self.vocab.analyze_coverage(text)

        assert isinstance(split_words, list)
        self.mock_tokenizer.tokenize.assert_called()

    def test_analyze_coverage_no_splits(self):
        """Test coverage analysis with no word splits"""
        # Mock tokenize method to return single tokens
        self.mock_tokenizer.tokenize.side_effect = lambda word: [word]

        text = "short words only"
        split_words = self.vocab.analyze_coverage(text)

        assert len(split_words) == 0

    @pytest.mark.parametrize(
        "input_text",
        [
            "",
            "simple text",
            "Text with UPPERCASE",
            "Text with numbers 123",
            "Hyphenated-word text",
        ],
    )
    def test_analyze_coverage_different_texts(self, input_text):
        """Test coverage analysis with different text patterns"""
        split_words = self.vocab.analyze_coverage(input_text)
        assert isinstance(split_words, list)

    def test_save_directory_creation(self, temp_dir):
        """Test that save creates directories if they don't exist"""
        nested_dir = temp_dir / "nested" / "path"
        with patch.object(self.mock_tokenizer, "save_pretrained"):
            self.vocab.save(nested_dir)
            assert nested_dir.exists()

    def test_pretokenizer_configuration(self):
        """Test that pretokenizer is properly configured"""
        tokenizer = self.vocab.get_tokenizer()
        assert tokenizer.pre_tokenizer is not None

    def test_normalizer_configuration(self):
        """Test that normalizer is properly configured"""
        tokenizer = self.vocab.get_tokenizer()
        assert tokenizer.normalizer is not None

    @pytest.mark.parametrize("tokenizer_type", list(TokenizerType))
    def test_get_tokenizer_all_types(self, tokenizer_type):
        """Test getting tokenizer for all tokenizer types"""
        self.vocab.tokenizer_type = tokenizer_type
        tokenizer = self.vocab.get_tokenizer()
        assert isinstance(tokenizer, Tokenizer)

    def test_error_handling(self, monkeypatch):
        """Test error handling when saving to invalid location"""

        def mock_save_pretrained(*args, **kwargs):
            raise Exception("save failed")

        monkeypatch.setattr(
            self.mock_tokenizer, "save_pretrained", mock_save_pretrained
        )
        with pytest.raises(Exception):
            self.vocab.save("/invalid/location/that/doesnt/exist")
