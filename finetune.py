import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from datasets import load_dataset

    books = load_dataset("parquet", data_files="training_set.parquet")
    books = books["train"].train_test_split(test_size=0.2)

    return (books,)


@app.cell
def _(books):
    books["train"][0]
    return


@app.cell
def _():
    from transformers import AutoTokenizer

    checkpoint = "cyberagent/CAT-Translate-0.8b"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    return checkpoint, tokenizer


@app.cell
def _(tokenizer):
    def preprocess_function(examples):
        max_length = 256
        input_ids_list = []
        labels_list = []
        for sentence, translation in zip(examples["sentence"], examples["translation"]):
            user_content = "Translate the following Japanese text into English.\n\n " + sentence
            messages = [{"role": "user", "content": user_content}]
            prompt_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=False
            )
            response_ids = tokenizer(translation, add_special_tokens=False)["input_ids"]
            response_ids = response_ids + [tokenizer.eos_token_id]
            input_ids = (prompt_ids + response_ids)[:max_length]
            labels = ([-100] * len(prompt_ids) + response_ids)[:max_length]
            input_ids_list.append(input_ids)
            labels_list.append(labels)
        return {"input_ids": input_ids_list, "labels": labels_list}


    return (preprocess_function,)


@app.cell
def _(books, preprocess_function):
    tokenized_books = books.map(
        preprocess_function,
        batched=True,
        remove_columns=books["train"].column_names,
    )
    tokenized_books

    return (tokenized_books,)


@app.cell
def _(tokenizer):
    import torch as _torch


    def data_collator(features):
        pad_id = tokenizer.pad_token_id
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        max_len = max(len(ids) for ids in input_ids)
        padded_ids, padded_mask, padded_labels = [], [], []
        for ids, lab in zip(input_ids, labels):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [pad_id] * pad_len)
            padded_mask.append([1] * len(ids) + [0] * pad_len)
            padded_labels.append(lab + [-100] * pad_len)
        return {
            "input_ids": _torch.tensor(padded_ids),
            "attention_mask": _torch.tensor(padded_mask),
            "labels": _torch.tensor(padded_labels),
        }


    return (data_collator,)


@app.cell
def _(checkpoint):
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype="bfloat16")

    return Trainer, TrainingArguments, model


@app.cell
def _():
    return


@app.cell
def _(
    Trainer,
    TrainingArguments,
    data_collator,
    model,
    tokenized_books,
    tokenizer,
):
    training_args = TrainingArguments(
        output_dir="cat-translate-jpen",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=2,
        fp16=False,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_books["train"],
        eval_dataset=tokenized_books["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
