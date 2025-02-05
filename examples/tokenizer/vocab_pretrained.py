from orbital.tokenizer import PretrainedVocabulary
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="bert-base-uncased", help="HuggingFace model name"
    )
    parser.add_argument(
        "--output", default="./custom_tokenizer", help="Output directory"
    )
    parser.add_argument(
        "--test_text",
        default="The quick brown fox jumps over the lazy dog.",
        help="Text to analyze coverage",
    )
    args = parser.parse_args()

    # Initialize vocabulary
    vocab = PretrainedVocabulary(args.model)

    # Save vocabulary and analyze coverage
    vocab.save(args.output)
    split_words = vocab.analyze_coverage(args.test_text)

    # Test tokenization
    tokenizer = vocab.get_tokenizer()
    encoded = tokenizer.encode(args.test_text)
    print(f"\nTest tokenization:")
    print(f"Input text: {args.test_text}")
    print(f"Encoded tokens: {encoded.tokens}")


if __name__ == "__main__":
    main()
