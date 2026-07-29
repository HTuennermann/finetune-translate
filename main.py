from typing import Any, Iterable


def format_sentence_card(row: dict[str, Any]) -> str:
    """Render a dataset row as a learner-friendly markdown card.

    The original Japanese sentence is shown as a heading (kept as-is), and
    everything else -- reading, translation, and essential vocabulary -- is
    formatted neatly below it.
    """
    sentence = (row.get("sentence") or "").strip()
    kana = (row.get("sentence_kana") or "").strip()
    translation = (row.get("translation") or "").strip()
    words = row.get("words") or []

    if not sentence:
        raise ValueError("row must contain a non-empty 'sentence'")

    lines: list[str] = []
    lines.append(f"## {sentence}")
    lines.append("")

    if kana:
        lines.append(f"**Reading:** {kana}")
        lines.append("")

    if translation:
        lines.append(f"**Translation:** {translation}")
        lines.append("")

    word_rows = [w for w in words if isinstance(w, dict)]
    if word_rows:
        lines.append("**Vocabulary:**")
        lines.append("")
        lines.append("| Word | Reading | Meaning |")
        lines.append("| :--- | :------ | :------ |")
        for w in word_rows:
            kanji = (w.get("kanji") or "").strip()
            reading = (w.get("kana") or "").strip()
            meaning = (w.get("translation") or "").strip()
            lines.append(f"| {kanji} | {reading} | {meaning} |")
        lines.append("")

    return "\n".join(lines)


def format_sentence_cards(rows: Iterable[dict[str, Any]]) -> str:
    """Format multiple rows into a single markdown string, separated by rules."""
    return "\n---\n\n".join(format_sentence_card(r) for r in rows)


if __name__ == "__main__":
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files="training_set.parquet", split="train")
    print(format_sentence_cards(ds[i] for i in (0, 5, 5000)))
