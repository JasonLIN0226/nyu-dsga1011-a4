"""Q4 statistics using T5 tokens, excluding special tokens and padding."""

import hashlib
import json
from pathlib import Path

from transformers import T5TokenizerFast


def statistics(path, tokenizer):
    texts = path.read_text(encoding="utf-8").splitlines()
    if not texts or any(not text.strip() for text in texts):
        raise ValueError(f"Empty dataset or blank example: {path}")
    tokens = tokenizer(
        texts, add_special_tokens=False, padding=False, truncation=False,
    )["input_ids"]
    return {
        "examples": len(texts),
        "mean_length": sum(map(len, tokens)) / len(tokens),
        "vocabulary_size": len({token for ids in tokens for token in ids}),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    folder = Path(__file__).resolve().parent
    tokenizer = T5TokenizerFast.from_pretrained(
        "google-t5/t5-small", cache_dir=str(folder.parent / ".cache/huggingface/hub"),
    )
    result = {
        "tokenizer": "google-t5/t5-small",
        "special_tokens_included": False,
        "preprocessing": "None; before and after statistics are identical.",
        "splits": {},
    }
    for split in ("train", "dev"):
        nl = statistics(folder / "data" / f"{split}.nl", tokenizer)
        sql = statistics(folder / "data" / f"{split}.sql", tokenizer)
        assert nl["examples"] == sql["examples"], "Unaligned input/target counts"
        result["splits"][split] = {"natural_language": nl, "sql": sql}
        print(f"\n{split.upper()}")
        print(f"Number of examples: {nl['examples']}")
        print(f"Mean sentence length: {nl['mean_length']:.2f}")
        print(f"Mean SQL query length: {sql['mean_length']:.2f}")
        print(f"Vocabulary size (natural language): {nl['vocabulary_size']}")
        print(f"Vocabulary size (SQL): {sql['vocabulary_size']}")
    output = folder / "results" / "q4_data_stats.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
