from datasets import load_dataset

from main import format_sentence_card

books = load_dataset("parquet", data_files="training_set.parquet")
books = books["train"].train_test_split(test_size=0.2)
books




from transformers import AutoTokenizer

checkpoint = "cyberagent/CAT-Translate-1.4b"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def preprocess_function(examples):
    max_length = 512
    input_ids_list = []
    labels_list = []
    for i in range(len(examples["sentence"])):
        row = {
            "sentence": examples["sentence"][i],
            "sentence_kana": examples["sentence_kana"][i],
            "translation": examples["translation"][i],
            "words": examples["words"][i],
        }
        sentence = row["sentence"]
        user_content = (
            "Produce a Japanese learner card (reading, translation, and "
            "essential vocabulary) for the following sentence.\n\n " + sentence
        )
        messages = [{"role": "user", "content": user_content}]
        prompt_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=False
        )
        card = format_sentence_card(row)
        response_ids = tokenizer(card, add_special_tokens=False)["input_ids"]
        response_ids = response_ids + [tokenizer.eos_token_id]
        input_ids = (prompt_ids + response_ids)[:max_length]
        labels = ([-100] * len(prompt_ids) + response_ids)[:max_length]
        input_ids_list.append(input_ids)
        labels_list.append(labels)
    return {"input_ids": input_ids_list, "labels": labels_list}




tokenized_books = books.map(
    preprocess_function,
    batched=True,
    remove_columns=books["train"].column_names,
)
tokenized_books


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




from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype="bfloat16")


training_args = TrainingArguments(
    output_dir="cat-translate-jpen-14",
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


