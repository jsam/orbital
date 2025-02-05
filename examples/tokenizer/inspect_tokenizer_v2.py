from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from collections import Counter


def analyze_model_vocabulary(model_name="bert-base-uncased"):
    """
    Analyze and demonstrate vocabulary manipulation for a pre-trained model
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Get the vocabulary
    vocab = tokenizer.get_vocab()

    # Basic vocabulary information
    print(f"Vocabulary size: {len(vocab)}")
    print(f"\nSpecial tokens:")
    for token in tokenizer.special_tokens_map.items():
        print(f"  {token[0]}: {token[1]}")

    # Sample some vocabulary entries
    print("\nSample vocabulary entries:")
    sample_tokens = list(vocab.items())[:5]
    for token, id in sample_tokens:
        print(f"  Token: '{token}', ID: {id}")

    # Demonstrate tokenization
    sample_text = "This is a sample sentence to analyze tokenization."
    tokens = tokenizer.tokenize(sample_text)
    token_ids = tokenizer.encode(sample_text)

    print(f"\nSample tokenization for: '{sample_text}'")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")

    # Demonstrate adding new tokens
    new_tokens = ["[NEW_TOKEN1]", "[NEW_TOKEN2]", "customword"]
    num_added = tokenizer.add_tokens(new_tokens)
    print(f"\nAdded {num_added} new tokens to vocabulary")
    print(f"New vocabulary size: {len(tokenizer.get_vocab())}")

    # Save expanded vocabulary
    expanded_vocab_file = "expanded_vocabulary.json"
    with open(expanded_vocab_file, "w") as f:
        json.dump(tokenizer.get_vocab(), f, indent=2)
    print(f"\nSaved expanded vocabulary to {expanded_vocab_file}")

    # Demonstrate how to resize model embeddings for new vocabulary
    print("\nExample of resizing model embeddings:")
    print("model = AutoModelForSequenceClassification.from_pretrained(model_name)")
    print("model.resize_token_embeddings(len(tokenizer))")

    return tokenizer


def analyze_text_for_vocab_expansion(text, tokenizer):
    """
    Analyze text to identify potential vocabulary additions
    """
    # Tokenize the text
    tokens = tokenizer.tokenize(text)

    # Find tokens that got split (potential candidates for vocabulary expansion)
    words = text.split()
    split_tokens = []

    for word in words:
        word_tokens = tokenizer.tokenize(word)
        if len(word_tokens) > 1:
            split_tokens.append((word, word_tokens))

    print("\nWords that got split into subwords:")
    for word, tokens in split_tokens:
        print(f"  '{word}' -> {tokens}")

    # Count token frequency
    token_counts = Counter(tokens)
    print("\nMost common tokens:")
    for token, count in token_counts.most_common(5):
        print(f"  '{token}': {count}")

    return split_tokens, token_counts


# Example usage
if __name__ == "__main__":
    # Initialize with base model
    tokenizer = analyze_model_vocabulary()

    # Analyze sample text for vocabulary expansion
    sample_text = """
    Machine learning models often need domain-specific vocabulary.
    Terms like tensorflow, pytorch, and keras might be split into subwords.
    We might want to add technical terms like backpropagation or hyperparameter.
    """

    split_tokens, token_counts = analyze_text_for_vocab_expansion(
        sample_text, tokenizer
    )

    # Demonstrate saving and loading modified tokenizer
    save_path = "expanded_tokenizer"
    tokenizer.save_pretrained(save_path)
    print(f"\nSaved expanded tokenizer to {save_path}")

    # Example of loading saved tokenizer
    print("To load the saved tokenizer:")
    print(f"loaded_tokenizer = AutoTokenizer.from_pretrained('{save_path}')")
