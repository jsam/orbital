import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass
from transformers import AutoTokenizer
from orbital.tokenizer.pretrained import TokenizerReader, TokenizerAggregator


def test_tokenization(tokenizer, text: str) -> Dict[str, Any]:
    encoded = tokenizer(text)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])

    return {
        "input_text": text,
        "input_ids": encoded["input_ids"],
        "tokens": tokens,
        "decoded_text": tokenizer.decode(encoded["input_ids"]),
    }


def print_analysis(model_name: str, test_text: str = "Hello, world! How are you?"):
    reader = TokenizerReader(model_name)
    analysis = reader.analyze_vocabulary()
    tokenization_test = test_tokenization(reader.tokenizer, test_text)

    print(f"=== Analysis for {model_name} ===\n")
    print(json.dumps(analysis, indent=2))
    print("\n=== Tokenization Test ===\n")
    print(json.dumps(tokenization_test, indent=2))


if __name__ == "__main__":
    # Example usage
    # print_analysis("microsoft/Phi-3.5-mini-instruct")
    models = [
        # SentencePiece unigram models
        "microsoft/phi-2",
        "microsoft/Phi-3.5-mini-instruct",
        "Qwen/Qwen2.5-Coder-3B-Instruct",
        "mistralai/Mistral-7B-v0.1",
        # BPE-based models
        "facebook/opt-125m",
        "THUDM/chatglm3-6b",
        "Deci/DeciLM-7b",
        # WordPiece models
        "google-bert/bert-base-uncased",
        # Different SentencePiece implementations
        "bigscience/bloom-560m",
    ]

    aggregator = TokenizerAggregator(
        model_names=models,
        output_dir=Path("tokenizer_configs"),
    )

    configs = aggregator.aggregate_configs()

    # Create summary YAML with high-level comparison
    summary = {
        model_name: {
            "vocab_size": config["analysis"]["vocab_size"],
            "tokenizer_type": config["analysis"]["tokenizer_type"],
            "model_max_length": config["analysis"]["model_max_length"],
            "vocabulary_stats": config["analysis"]["vocabulary_stats"],
        }
        for model_name, config in configs.items()
    }

    with open("tokenizer_configs/summary.yaml", "w", encoding="utf-8") as f:
        yaml.dump(summary, f, allow_unicode=True, sort_keys=False)
