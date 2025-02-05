from transformers import AutoTokenizer
import json
from pathlib import Path
import difflib


def load_and_verify_expanded_tokenizer(
    expanded_tokenizer_path="expanded_tokenizer",
    original_model_name="bert-base-uncased",
    vocab_file="expanded_vocabulary.json",
):
    """
    Load an expanded tokenizer and verify its contents against the original

    Args:
        expanded_tokenizer_path: Path to the saved expanded tokenizer
        original_model_name: Name of the original model for comparison
        vocab_file: Path to the saved vocabulary JSON file
    """
    # Load both tokenizers
    print(f"Loading expanded tokenizer from {expanded_tokenizer_path}")
    expanded_tokenizer = AutoTokenizer.from_pretrained(expanded_tokenizer_path)

    print(f"Loading original tokenizer from {original_model_name}")
    original_tokenizer = AutoTokenizer.from_pretrained(original_model_name)

    # Compare vocabulary sizes
    expanded_vocab = expanded_tokenizer.get_vocab()
    original_vocab = original_tokenizer.get_vocab()

    print("\nVocabulary comparison:")
    print(f"Original vocabulary size: {len(original_vocab)}")
    print(f"Expanded vocabulary size: {len(expanded_vocab)}")

    # Find new tokens
    new_tokens = set(expanded_vocab.keys()) - set(original_vocab.keys())
    print(f"\nNew tokens added ({len(new_tokens)}):")
    for token in sorted(new_tokens):
        print(f"  '{token}' (ID: {expanded_vocab[token]})")

    # Load and verify against saved vocabulary file
    if Path(vocab_file).exists():
        with open(vocab_file, "r") as f:
            saved_vocab = json.load(f)

        print("\nVerifying against saved vocabulary file:")
        if saved_vocab == expanded_vocab:
            print("✓ Loaded tokenizer vocabulary matches saved vocabulary file")
        else:
            print("! Discrepancy found between loaded tokenizer and saved vocabulary")
            # Find differences
            vocab_diff = set(saved_vocab.items()) ^ set(expanded_vocab.items())
            if vocab_diff:
                print("Differences found:")
                for item in vocab_diff:
                    print(f"  {item}")

    # Demonstrate tokenization with new vocabulary
    test_sentences = [
        "This is a basic sentence.",
        "Let's test some technical terms: tensorflow keras pytorch",
        "Testing special tokens and new additions",
    ]

    print("\nTokenization comparison:")
    for sentence in test_sentences:
        print(f"\nInput: {sentence}")

        # Original tokenization
        orig_tokens = original_tokenizer.tokenize(sentence)
        print("Original tokenization:", orig_tokens)

        # New tokenization
        new_tokens = expanded_tokenizer.tokenize(sentence)
        print("Expanded tokenization:", new_tokens)

        # Show differences
        if orig_tokens != new_tokens:
            print("Differences in tokenization:")
            for diff in difflib.ndiff(orig_tokens, new_tokens):
                if diff[0] != " ":  # Show only changes
                    print(f"  {diff}")

    return expanded_tokenizer


def test_custom_tokens(tokenizer, custom_text):
    """
    Test how the tokenizer handles specific text with custom tokens

    Args:
        tokenizer: The loaded tokenizer
        custom_text: Text to tokenize and analyze
    """
    print(f"\nTesting custom text: '{custom_text}'")

    # Tokenize
    tokens = tokenizer.tokenize(custom_text)
    token_ids = tokenizer.encode(custom_text)

    # Show results
    print("Tokens:", tokens)
    print("Token IDs:", token_ids)

    # Verify roundtrip
    decoded = tokenizer.decode(token_ids)
    print("Decoded:", decoded)

    # Check if any tokens got split unexpectedly
    words = custom_text.split()
    for word in words:
        word_tokens = tokenizer.tokenize(word)
        if len(word_tokens) > 1:
            print(f"Note: '{word}' is still being split into {word_tokens}")


if __name__ == "__main__":
    # Load the expanded tokenizer
    expanded_tokenizer = load_and_verify_expanded_tokenizer()

    # Test with custom text containing new tokens
    test_text = "Using tensorflow and pytorch for backpropagation training"
    test_custom_tokens(expanded_tokenizer, test_text)

    # Test with special tokens
    special_text = "[NEW_TOKEN1] This is a test with [NEW_TOKEN2] and customword"
    test_custom_tokens(expanded_tokenizer, special_text)
