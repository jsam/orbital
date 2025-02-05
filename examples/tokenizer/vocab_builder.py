from orbital.tokenizer.builder import VocabularyBuilder
from orbital.tokenizer.config.dataclasses import TokenizerType


def test_tokenizer(tokenizer, test_texts):
    """Test the tokenizer on sample texts"""
    print("\nTesting tokenizer:")

    for text in test_texts:
        print(f"\nInput text: {text}")

        # Tokenize
        encoding = tokenizer.encode(text)

        print(f"Tokens: {encoding.tokens}")
        print(f"IDs: {encoding.ids}")

        # Verify round-trip
        decoded = tokenizer.decode(encoding.ids)
        print(f"Decoded: {decoded}")


# Example usage
if __name__ == "__main__":
    # Test with different tokenizer types
    for tokenizer_type in TokenizerType:
        print(f"\nTesting {tokenizer_type.value} tokenizer")

        builder = VocabularyBuilder(
            tokenizer_type=tokenizer_type,
            vocab_size=30000,
            min_frequency=2,
            special_tokens=[
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "[BOS]",
                "[EOS]",
            ],
        )

        test_texts = [
            "This is a simple test sentence.",
            "Let's see how it handles technical terms and rare words.",
            "Testing special tokens and boundaries.",
        ]

        # Analyze text files and train tokenizer
        builder.analyze_texts(test_texts)

        # Save tokenizer
        output_dir = f"custom_tokenizer_{tokenizer_type.value}"
        tokenizer = builder.build_tokenizer(output_dir=output_dir)

        # Test the tokenizer
        test_tokenizer(tokenizer, test_texts)
