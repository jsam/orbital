import pytest
from pathlib import Path
import shutil
import json
from unittest.mock import Mock, patch, MagicMock
from tokenizers import Tokenizer

from orbital.tokenizer import (
    VocabularyBuilder,
    TokenizerType,
)  # Update with your actual module


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files"""
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()
    yield test_dir
    # Cleanup
    if test_dir.exists():
        shutil.rmtree(test_dir)


@pytest.fixture
def sample_texts():
    """Provide sample texts for testing"""
    return [
        "This is a test document.",
        "Another test document with more words.",
        "Technical terms like tensorflow and pytorch.",
    ]


@pytest.fixture
def sample_txt_files(temp_dir):
    """Create sample text files for testing"""
    files = []
    for i, text in enumerate(
        [
            "First test file content.",
            "Second test file with different content.",
            "Third test file with more text.",
        ]
    ):
        file_path = temp_dir / f"test_{i}.txt"
        file_path.write_text(text)
        files.append(file_path)
    return files


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing"""
    mock = Mock(spec=Tokenizer)
    mock.get_vocab.return_value = {"test": 0, "word": 1}
    return mock


class TestVocabularyBuilder:
    @pytest.fixture(autouse=True)
    def setup(self, temp_dir):
        """Setup test environment before each test"""
        self.output_dir = temp_dir / "output"
        self.builder = VocabularyBuilder(
            tokenizer_type=TokenizerType.WORDPIECE, vocab_size=1000, min_frequency=1
        )
        yield
        # Cleanup after each test
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    @pytest.mark.parametrize("tokenizer_type", list(TokenizerType))
    def test_initialization(self, tokenizer_type):
        """Test initialization with different tokenizer types"""
        builder = VocabularyBuilder(
            tokenizer_type=tokenizer_type, vocab_size=1000, min_frequency=1
        )
        assert builder.tokenizer_type == tokenizer_type
        assert builder.vocab_size == 1000
        assert builder.min_frequency == 1
        assert isinstance(builder.tokenizer, Tokenizer)

    def test_analyze_texts(self, sample_texts):
        """Test analyzing list of text strings"""
        with patch.object(self.builder.tokenizer, "train_from_iterator") as mock_train:
            self.builder.analyze_texts(sample_texts)
            mock_train.assert_called_once()
            # Verify word counter was updated
            assert len(self.builder.word_counter) > 0
            assert self.builder.word_counter["test"] == 2  # 'test' appears twice

    def test_analyze_texts_empty_list(self):
        """Test analyzing empty text list"""
        with pytest.raises(ValueError, match="Empty text list provided"):
            self.builder.analyze_texts([])

    def test_analyze_directory(self, temp_dir, sample_txt_files):
        """Test analyzing directory of text files"""
        with patch.object(self.builder.tokenizer, "train_from_iterator") as mock_train:
            self.builder.analyze_directory(temp_dir)
            mock_train.assert_called_once()
            assert len(self.builder.word_counter) > 0

    def test_analyze_directory_not_found(self):
        """Test analyzing non-existent directory"""
        with pytest.raises(ValueError, match="Directory not found"):
            self.builder.analyze_directory(Path("nonexistent_dir"))

    def test_analyze_files(self, sample_txt_files):
        """Test analyzing list of file paths"""
        with patch.object(self.builder.tokenizer, "train_from_iterator") as mock_train:
            self.builder.analyze_files(sample_txt_files)
            mock_train.assert_called_once()
            assert len(self.builder.word_counter) > 0

    def test_analyze_files_empty_list(self):
        """Test analyzing empty file list"""
        with pytest.raises(ValueError, match="Empty file list provided"):
            self.builder.analyze_files([])

    def test_analyze_files_nonexistent(self, temp_dir):
        """Test analyzing non-existent files"""
        non_existent = [temp_dir / "nonexistent.txt"]
        self.builder.analyze_files(non_existent)  # Should handle gracefully
        assert len(self.builder.word_counter) == 0

    @pytest.mark.parametrize(
        "special_tokens", [None, ["[PAD]", "[UNK]"], ["[CUSTOM1]", "[CUSTOM2]"]]
    )
    def test_special_tokens(self, special_tokens):
        """Test initialization with different special tokens"""
        builder = VocabularyBuilder(
            tokenizer_type=TokenizerType.WORDPIECE,
            vocab_size=1000,
            special_tokens=special_tokens,
        )
        assert isinstance(builder.special_tokens, list)
        if special_tokens:
            assert all(token in builder.special_tokens for token in special_tokens)

    def test_build_tokenizer(self, mock_tokenizer, temp_dir):
        """Test building and saving tokenizer"""
        self.builder.tokenizer = mock_tokenizer

        # Mock the save method
        with patch.object(mock_tokenizer, "save"):
            tokenizer = self.builder.build_tokenizer(temp_dir / "output")

            # Verify files were created
            assert (temp_dir / "output").exists()
            assert (temp_dir / "output" / "config.json").exists()
            assert (temp_dir / "output" / "vocab.json").exists()

            # Verify config contents
            with open(temp_dir / "output" / "config.json") as f:
                config = json.load(f)
                assert config["tokenizer_type"] == TokenizerType.WORDPIECE.value
                assert config["vocab_size"] == 1000

    def test_analyze_coverage(self, mock_tokenizer):
        """Test vocabulary coverage analysis"""
        self.builder.tokenizer = mock_tokenizer
        mock_tokenizer.encode.return_value = MagicMock(tokens=["test", "##ing"])

        text = "testing word coverage"
        split_words = self.builder.analyze_coverage(text)

        assert isinstance(split_words, list)
        mock_tokenizer.encode.assert_called()

    def test_read_file_encoding_error(self, temp_dir):
        """Test handling of file reading errors"""
        # Create a file with invalid encoding
        file_path = temp_dir / "invalid.txt"
        with open(file_path, "wb") as f:
            f.write(b"\x80\x81")  # Invalid UTF-8

        result = self.builder._read_file(file_path)
        assert result == []  # Should return empty list on error

    @pytest.mark.parametrize(
        "file_content,expected_words",
        [
            ("simple text", 2),
            ("text with numbers 123", 4),
            ("UPPERCASE text", 2),
            ("hyphenated-word text", 3),
            ("", 0),
        ],
    )
    def test_word_counting(self, temp_dir, file_content, expected_words):
        """Test word counting with different text patterns"""
        file_path = temp_dir / "test.txt"
        file_path.write_text(file_content)

        self.builder._read_file(file_path)
        assert sum(self.builder.word_counter.values()) == expected_words

    def test_pretokenizer_configuration(self):
        """Test that pre-tokenizer is properly configured for each tokenizer type"""
        for tokenizer_type in TokenizerType:
            builder = VocabularyBuilder(tokenizer_type=tokenizer_type)
            assert builder.tokenizer.pre_tokenizer is not None

    def test_method_chaining(self, sample_texts):
        """Test that analysis methods support method chaining"""
        with patch.object(self.builder.tokenizer, "train_from_iterator"):
            result = self.builder.analyze_texts(sample_texts)
            assert isinstance(result, VocabularyBuilder)
