from typing import List


class MockTokenizer:
    def tokenize(self, text: str) -> List[str]:
        return ["test", "token"]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return [1, 2, 3]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return "test token"

    def add_tokens(self, tokens: List[str]) -> int:
        return len(tokens)

    def get_vocab(self) -> dict:
        return {"test": 1, "token": 2}


class MockSentencePieceTokenizer(MockTokenizer):
    spiece_model = True
