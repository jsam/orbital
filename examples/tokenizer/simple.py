from orbital.tokenizer import Tokenizer


def main():
    # Initialize tokenizer with a commonly used model
    # You can use any model supported by HuggingFace's AutoTokenizer
    model_name = "microsoft/phi-2"  # or "mistralai/Mistral-7B-v0.1" or other models
    tokenizer = Tokenizer(model_name)

    # Example 1: Basic tokenization
    text = "Hello, let's test this tokenizer!"
    print("\nExample 1: Basic tokenization")
    print(f"Original text: {text}")
    tokens = tokenizer.tokenize(text)
    print(f"Tokens: {tokens}")

    # Example 2: Encoding to token IDs
    print("\nExample 2: Encoding to token IDs")
    token_ids = tokenizer.encode(text)
    print(f"Token IDs: {token_ids}")

    # Example 3: Decoding back to text
    print("\nExample 3: Decoding back to text")
    decoded_text = tokenizer.decode(token_ids)
    print(f"Decoded text: {decoded_text}")

    # Example 4: Working with special tokens
    print("\nExample 4: Special tokens handling")
    text_pair = ["Hello", "world!"]
    # Encode with special tokens (like [CLS], [SEP], etc. depending on the model)
    token_ids_with_special = tokenizer.encode(
        " ".join(text_pair), add_special_tokens=True
    )
    token_ids_without_special = tokenizer.encode(
        " ".join(text_pair), add_special_tokens=False
    )
    print(f"With special tokens: {token_ids_with_special}")
    print(f"Without special tokens: {token_ids_without_special}")

    # Example 5: Vocabulary inspection
    print("\nExample 5: Vocabulary inspection")
    vocab = tokenizer.get_vocab()
    print(f"Vocabulary size: {len(vocab)}")
    # Print first 5 tokens and their IDs
    print("First 5 tokens in vocabulary:")
    for token, id_ in list(vocab.items())[:5]:
        print(f"Token: {token}, ID: {id_}")

    # Example 6: Adding new tokens
    print("\nExample 6: Adding new tokens")
    new_tokens = ["<USER>", "<SYSTEM>"]
    num_added = tokenizer.add_tokens(new_tokens)
    print(f"Number of tokens added: {num_added}")

    # Test the newly added tokens
    if num_added > 0:
        test_text = "Hello <USER>, how are you?"
        tokens = tokenizer.tokenize(test_text)
        print(f"Tokenized text with new token: {tokens}")


if __name__ == "__main__":
    main()
