import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import glob 

    return glob, mo


@app.cell
def _(glob):
    glob.glob("Japanese*")
    return


@app.cell
def _(mo):
    _df = mo.sql(
        f"""
        ATTACH IF NOT EXISTS 'Japanese_Core_1000_Vocab__Pitch_Accent/collection.anki2' as c1000 (TYPE SQLITE, READ_ONLY);
        ATTACH IF NOT EXISTS 'Japanese_Core_2000_Vocab__Pitch_Accent/collection.anki2' as c2000 (TYPE SQLITE, READ_ONLY);
        ATTACH IF NOT EXISTS 'Japanese_Core_3000_Vocab__Pitch_Accent/collection.anki2' as c3000 (TYPE SQLITE, READ_ONLY);
        ATTACH IF NOT EXISTS 'Japanese_Core_4000_Vocab__Pitch_Accent/collection.anki2' as c4000 (TYPE SQLITE, READ_ONLY);
        ATTACH IF NOT EXISTS 'Japanese_Core_5000_Vocab__Pitch_Accent/collection.anki2' as c5000 (TYPE SQLITE, READ_ONLY);
        ATTACH IF NOT EXISTS 'Japanese_Core_6000_Vocab__Pitch_Accent/collection.anki2' as c6000 (TYPE SQLITE, READ_ONLY);
        SELECT table_name FROM INFORMATION_SCHEMA.TABLES where table_catalog == 'c6000';
        """
    )
    return


@app.cell
def _():
    5
    return


@app.cell
def _(mo):
    df = mo.sql(
        f"""
        SELECT * FROM c1000.notes
        UNION ALL
        SELECT * FROM c2000.notes
        UNION ALL
        SELECT * FROM c3000.notes
        UNION ALL
        SELECT * FROM c4000.notes
        UNION ALL
        SELECT * FROM c5000.notes
        UNION ALL
        SELECT * FROM c6000.notes
        """
    )
    return (df,)


@app.cell
def _(df):
    data = df["flds"].str.split("\x1f")
    return (data,)


@app.cell
def _(data):
    import polars as pl

    vocab = (
        pl.DataFrame({"flds": data})
        .with_columns(
            pl.col("flds").list.to_struct(
                fields=[
                    "Index",
                    "Word",
                    "Transliteration",
                    "Meaning",
                    "Part of Speech",
                    "Example Sentence",
                    "Sentence Transliteration",
                    "Sentence Translation",
                    "Word Audio",
                    "Sentence Audio",
                    "Pitch Accent URL",
                    "Pitch Accent",
                ]
            )
        )
        .unnest("flds")
    )
    vocab
    return pl, vocab


@app.cell
def _(vocab):
    import os

    _vocab_path = "vocab.parquet"
    vocab.write_parquet(_vocab_path)
    f"{os.path.getsize(_vocab_path) / 1024:.0f} KB written to {_vocab_path}"

    return (os,)


@app.cell
def _(pl, vocab):
    examples = (
        vocab.drop("Word Audio", "Sentence Audio")
        .with_columns(
            pl.col("Example Sentence").str.split("<br><br>"),
            pl.col("Sentence Transliteration").str.split("<br><br>"),
            pl.col("Sentence Translation").str.split("<br><br>"),
        )
        .explode("Example Sentence", "Sentence Transliteration", "Sentence Translation")
        .with_columns(example_id=pl.int_range(1, pl.len() + 1).over("Index"))
        .select("Index", "example_id", pl.exclude("Index", "example_id"))
    )
    examples
    return (examples,)


@app.cell
def _(examples, os):
    _examples_path = "examples.parquet"
    examples.write_parquet(_examples_path)
    f"{os.path.getsize(_examples_path) / 1024:.0f} KB written to {_examples_path}"
    return


@app.cell
def _():
    return


@app.cell
def _(os):
    from openai import AsyncOpenAI
    import json

    _DEEPSEEK_CLIENT = AsyncOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com/v1",
    )
    _LLM_MODEL = "deepseek-chat"

    _SYSTEM = """You are a precise Japanese lexical analyzer.

    Given a Japanese sentence and its kana reading, output every CONTENT vocabulary word in its dictionary base form (辞書形 / 終止形). For each word return an object with:
    - "kanji": the written form of the base word AS IT IS WRITTEN in the sentence
    - "kana": the full hiragana reading of the base-form word
    - "translation": one concise English gloss

    KANJI FIELD RULE (critical):
    - MIRROR the script used in the sentence: if the sentence writes the word in kanji, output that kanji (lemmatized to base form); if the sentence writes it in kana, output the kana.
    - NEVER invent a kanji spelling that does not appear in the sentence and is not an unquestionably standard, everyday modern spelling.
    - NEVER output archaic/literary/ateji spellings: 此れ(これ), 其れ/其(それ), 彼 for あれ, 貴方(あなた), 此方(こちら), 何時(いつ), 或る(ある). For those, use the kana.

    EXCLUDE entirely (never list):
    - Pronouns & demonstratives: 私 僕 俺 あなた 君 彼 彼女 これ それ あれ この その あの ここ そこ あそこ こちら そちら あちら だれ どれ どこ いつ なに 何.
    - Pure grammar: particles (は が で に を と も へ や より から まで の), copulas (だ です である), politeness/auxiliary verbs (ます ている たい ない れる られる た て), conjunctions, filler, and isolated digits/numbers.

    INCLUDE (base form): nouns, suru-verbs, verbs, i-adjectives, na-adjectives, adverbs, and fixed expressions.

    LEMMA RULES:
    - Lemmatize all conjugated/inflected forms to the base form. Examples: 行った->行く, 降っている->降る, すごくて->凄い, 食べなかった->食べる, 紹介しています->紹介する.
    - List each distinct base word only ONCE.

    OUTPUT: ONLY a JSON object {"words": [{"kanji": "...", "kana": "...", "translation": "..."}, ...]}. No markdown, no commentary."""


    # archaic/ateji spellings that are never correct modern forms -> replace with kana
    _ATEJI = {"此れ", "其れ", "其", "貴方", "此方", "何時", "或る"}


    def _scrub(words):
        for _w in words:
            if _w.get("kanji") in _ATEJI:
                _w["kanji"] = _w.get("kana", _w.get("kanji"))
        return words


    async def annotate(sentence, kana):
        resp = await _DEEPSEEK_CLIENT.chat.completions.create(
            model=_LLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"Sentence: {sentence}\nKana: {kana}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=800,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines)
        try:
            return _scrub(json.loads(raw).get("words", []))
        except Exception:
            return []

    return annotate, json


@app.cell
def _(annotate, examples, json, mo):
    import asyncio

    _ANN_CACHE = "annotations_cache.jsonl"


    def _load_ann_cache():
        _done = {}
        try:
            with open(_ANN_CACHE, encoding="utf-8") as _fh:
                for _line in _fh:
                    _line = _line.strip()
                    if not _line:
                        continue
                    try:
                        _rec = json.loads(_line)
                    except Exception:
                        continue
                    _done[(_rec["Index"], _rec["example_id"])] = _rec
        except FileNotFoundError:
            pass
        return _done


    _ann_total = examples.height
    _ann_bar = mo.status.progress_bar(total=_ann_total, title="Annotating sentences", subtitle="starting")


    async def _ann_one(_row, _sem):
        _words = []
        for _attempt in range(5):
            try:
                async with _sem:
                    _words = await annotate(_row["Example Sentence"], _row["Sentence Transliteration"])
                break
            except Exception:
                await asyncio.sleep(min(2 ** _attempt, 20))
        return {
            "Index": _row["Index"],
            "example_id": _row["example_id"],
            "sentence": _row["Example Sentence"],
            "sentence_kana": _row["Sentence Transliteration"],
            "words": _words,
        }


    async def _annotate_all():
        _rows = examples.select("Index", "example_id", "Example Sentence", "Sentence Transliteration").to_dicts()
        _done = _load_ann_cache()
        _todo = [_r for _r in _rows if (_r["Index"], _r["example_id"]) not in _done]
        _sem = asyncio.Semaphore(10)
        _n_done = len(_done)
        _fh = open(_ANN_CACHE, "a", encoding="utf-8")

        async def _wrapped(_r):
            nonlocal _n_done
            _rec = await _ann_one(_r, _sem)
            _fh.write(json.dumps(_rec, ensure_ascii=False) + "\n")
            _fh.flush()
            _n_done += 1
            try:
                _ann_bar.progress = _n_done
                _ann_bar.subtitle = f"{_n_done}/{_ann_total}"
            except Exception:
                pass

        await asyncio.gather(*(_wrapped(_r) for _r in _todo))
        _fh.close()
        try:
            _ann_bar.progress = _ann_total
            _ann_bar.subtitle = "done"
        except Exception:
            pass
        return _n_done


    _annotation_task = asyncio.create_task(_annotate_all())
    _ann_bar
    return


@app.cell
def _(json, pl):
    _ann_records = []
    _ann_seen = set()
    with open("annotations_cache.jsonl", encoding="utf-8") as _fh:
        for _line in _fh:
            _rec = json.loads(_line)
            _key = (_rec["Index"], _rec["example_id"])
            if _key in _ann_seen:
                for _i, _r in enumerate(_ann_records):
                    if (_r["Index"], _r["example_id"]) == _key:
                        _ann_records[_i] = _rec
                        break
            else:
                _ann_seen.add(_key)
                _ann_records.append(_rec)

    sentence_annotations = (
        pl.DataFrame(_ann_records)
        .with_columns(_idx=pl.col("Index").str.to_integer())
        .sort("_idx", "example_id")
        .drop("_idx")
    )
    sentence_annotations.write_parquet("sentence_annotations.parquet")

    _ann_empty = sentence_annotations.filter(pl.col("words").list.len() == 0).height
    _ann_words = sentence_annotations.select(pl.col("words").list.len().sum()).item()
    f"{len(sentence_annotations)} rows ({_ann_empty} empty) | {_ann_words} word entries -> sentence_annotations.parquet"
    return (sentence_annotations,)


@app.cell
def _(examples, pl, sentence_annotations):
    import random
    import string

    _hiragana = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
    _katakana = "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"
    _noise_pool = string.ascii_letters + string.digits + "!@#$%&*?=-+" + _hiragana + _katakana

    _rng = random.Random(20240723)
    _P_NOISE = 0.30


    def _noisy(word):
        if _rng.random() < _P_NOISE:
            _ch = _rng.choice(_noise_pool)
            return (_ch + word) if _rng.random() < 0.5 else (word + _ch)
        return word


    _vocab_src = pl.read_parquet("vocab.parquet").select("Index", "Word", "Transliteration", "Meaning")
    _word_rows = []
    _n_noisy = 0
    for _r in _vocab_src.to_dicts():
        _w = _r["Word"]
        _noised = _noisy(_w)
        if _noised != _w:
            _n_noisy += 1
        _word_rows.append({
            "Index": _r["Index"],
            "example_id": 0,
            "sentence": _noised,
            "sentence_kana": _r["Transliteration"],
            "words": [{"kanji": _w, "kana": _r["Transliteration"], "translation": _r["Meaning"]}],
        })

    word_annotations = (
        pl.DataFrame(_word_rows)
        .with_columns(_idx=pl.col("Index").str.to_integer())
        .sort("_idx")
        .drop("_idx")
    )
    word_annotations.write_parquet("word_annotations.parquet")

    # full translation: deck sentence translation for sentences; word meaning for single words
    _tsent = examples.select("Index", "example_id", pl.col("Sentence Translation").alias("translation"))
    training_set = (
        pl.concat(
            [
                sentence_annotations.join(_tsent, on=["Index", "example_id"], how="left"),
                word_annotations.with_columns(
                    translation=pl.col("words").list.get(0).struct.field("translation")
                ),
            ],
            how="vertical_relaxed",
        )
        .with_columns(_idx=pl.col("Index").str.to_integer())
        .sort("_idx", "example_id")
        .drop("_idx")
    )
    training_set.write_parquet("training_set.parquet")

    f"word_annotations: {len(word_annotations)} ({_n_noisy} noisy) | training_set: {len(training_set)} rows, +translation (nulls: {training_set['translation'].null_count()})"
    return (training_set,)


@app.cell
async def _(annotate, examples, mo, pl):
    _pilot_rows = examples.head(20).select(
        "Index", "example_id", "Example Sentence", "Sentence Transliteration"
    ).to_dicts()

    _pilot_results = []
    for _r in _pilot_rows:
        _w = await annotate(_r["Example Sentence"], _r["Sentence Transliteration"])
        _pilot_results.append({
            "Index": _r["Index"],
            "example_id": _r["example_id"],
            "sentence": _r["Example Sentence"],
            "sentence_kana": _r["Sentence Transliteration"],
            "n_words": len(_w),
            "words": _w,
        })

    for _r in _pilot_results[:5]:
        print(_r["sentence"])
        print("  kana:", _r["sentence_kana"])
        for _w in _r["words"]:
            print(f"   - {_w.get('kanji')} | {_w.get('kana')} | {_w.get('translation')}")
        print()

    mo.ui.table(pl.DataFrame(_pilot_results))
    return


@app.cell
def _(training_set):
    training_set
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
