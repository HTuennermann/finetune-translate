import argparse
import glob
import os
import re

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIRS = {
    "0.8b (finetuned)": "cat-translate-jpen",
    "1.4b (finetuned)": "cat-translate-jpen-14",
}

BASE_CHECKPOINTS = {
    "0.8b (base)": "cyberagent/CAT-Translate-0.8b",
    "1.4b (base)": "cyberagent/CAT-Translate-1.4b",
}

DEFAULT_SENTENCES = [
        "私は毎日日本語を勉強しています。",
        "東京タワーはとても高いです。",
        "猫がソファの上で寝ている。",
        "明日、友達と一緒に映画を見に行きます。",
        "このレストランの料理は美味しいですね。",
    ]


def latest_checkpoint(model_dir):
    checkpoints = glob.glob(os.path.join(model_dir, "checkpoint-*"))
    if not checkpoints:
        return model_dir
    def step(path):
        m = re.search(r"checkpoint-(\d+)", path)
        return int(m.group(1)) if m else -1
    return max(checkpoints, key=step)


def load_model(path, device):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype="bfloat16").to(device)
    model.eval()
    return tokenizer, model


def translate(sentence, tokenizer, model, device, max_new_tokens=128):
    user_content = "Translate the following Japanese text into English.\n\n " + sentence
    messages = [{"role": "user", "content": user_content}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=False
    )
    input_ids = torch.tensor([prompt_ids], device=device)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output_ids[0, input_ids.shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text


def load_examples(parquet_path, n):
    df = pd.read_parquet(parquet_path)
    sample = df.dropna(subset=["sentence", "translation"]).sample(
        n=n, random_state=42
    )
    return list(sample["sentence"]), list(sample["translation"])


def main():
    parser = argparse.ArgumentParser(description="Demo finetuned JP->EN translators")
    parser.add_argument("--parquet", default="training_set.parquet")
    parser.add_argument("--n", type=int, default=5, help="number of held-out samples to show")
    parser.add_argument("--no-base", action="store_true", help="skip loading base models")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    sources = []
    references = []
    if os.path.exists(args.parquet):
        sources, references = load_examples(args.parquet, args.n)
        for s in DEFAULT_SENTENCES:
            sources.append(s)
            references.append(None)
    else:
        for s in DEFAULT_SENTENCES:
            sources.append(s)
            references.append(None)

    targets = {}
    for name, model_dir in MODEL_DIRS.items():
        targets[name] = latest_checkpoint(model_dir)
    if not args.no_base:
        for name, ckpt in BASE_CHECKPOINTS.items():
            targets[name] = ckpt

    results = {name: [] for name in targets}
    print(f"\nLoading and running {len(targets)} model(s) on {len(sources)} sentence(s)...\n")

    for name, path in targets.items():
        print(f"  -> loading {name} from {path}")
        tokenizer, model = load_model(path, args.device)
        for sentence in sources:
            results[name].append(translate(sentence, tokenizer, model, args.device))
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("EXAMPLE RESULTS")
    print("=" * 80)
    for i, sentence in enumerate(sources):
        ref = references[i] if i < len(references) else None
        print(f"\n[{i + 1}] JP:  {sentence}")
        if ref:
            print(f"    REF: {ref}")
        for name in targets:
            print(f"    {name}: {results[name][i]}")


if __name__ == "__main__":
    main()
