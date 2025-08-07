import re
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report


LABEL_COLS = ("findings", "clinicaldata", "ExamName", "impression")

def extract_sections(row, text_col="ReportText", label_cols=LABEL_COLS, sep="\n\n"):
    out = {}
    out[text_col] = row[text_col]
    for col in label_cols:
        raw = row.get(col, "") or ""
        parts = [blk.strip() for blk in raw.split(sep) if blk.strip()]
        out[col] = parts
    return out

def chunk_labels_for_row(report_tokens, sections, label_cols=LABEL_COLS):
    N = len(report_tokens)
    chunk_labels = [None] * N
    used = set()

    for sec in label_cols:
        # Sort blocks by length descending to match longer chunks first
        for block in sorted(sections[sec], key=lambda b: -len(b.split())):
            blk_toks = block.split()
            L = len(blk_toks)
            if L == 0:
                continue
            for i in range(0, N - L + 1):
                if any((i + k) in used for k in range(L)):
                    continue
                if report_tokens[i:i + L] == blk_toks:
                    for k in range(L):
                        chunk_labels[i + k] = sec
                    used.update(range(i, i + L))
    return chunk_labels

def bioes_from_chunks(chunk_labels):
    N = len(chunk_labels)
    tags = ["O"] * N
    i = 0
    while i < N:
        sec = chunk_labels[i]
        if sec is None:
            tags[i] = "O"
            i += 1
        else:
            j = i + 1
            while j < N and chunk_labels[j] == sec:
                j += 1
            length = j - i
            if length == 1:
                tags[i] = f"S-{sec}"
            elif length == 2:
                tags[i] = f"B-{sec}"
                tags[i + 1] = f"E-{sec}"
            else:
                tags[i] = f"B-{sec}"
                for k in range(i + 1, j - 1):
                    tags[k] = f"I-{sec}"
                tags[j - 1] = f"E-{sec}"
            i = j
    return tags

def run_finetuning(df: pd.DataFrame, model_name="emilyalsentzer/Bio_ClinicalBERT"):
    # 1) Extract sections
    df["sections"] = df.apply(extract_sections, axis=1)

    # 2) Tokenize full report
    df["tokens"] = df["sections"].apply(lambda s: s["ReportText"].split())

    # 3) Chunk tokens to section labels
    df["chunk_labels"] = df.apply(
        lambda row: chunk_labels_for_row(
            report_tokens=row["tokens"],
            sections=row["sections"],
            label_cols=LABEL_COLS
        ),
        axis=1
    )

    # 4) Convert chunks to BIOES tags
    df["labels"] = df["chunk_labels"].apply(bioes_from_chunks)

    # 5) Remove rows containing "O" tags (optional: keep only fully labeled data)
    mask = df["labels"].apply(lambda tags: "O" in tags)
    df_clean = df[~mask].reset_index(drop=True)

    # 6) Train/validation/test split
    train_df, temp_df = train_test_split(df_clean, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df)
    })

    # 7) Tokenizer and label map
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 512

    label_list = sorted({lab for row in df["labels"] for lab in row})
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    # 8) Tokenize and align labels for each example
    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
        )
        word_ids = tokenized.word_ids()
        labels = []
        for wid in word_ids:
            if wid is None:
                labels.append(-100)
            else:
                labels.append(label2id[examples["labels"][wid]])
        tokenized["labels"] = labels
        return tokenized

    tokenized_datasets = dataset_dict.map(
        tokenize_and_align,
        batched=False,
        remove_columns=dataset_dict["train"].column_names
    )

    # 9) Load model
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    # 10) Training arguments and trainer setup
    training_args = TrainingArguments(
        output_dir="ner_model",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        report_to="none"
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # 11) Train model
    trainer.train()

    # 12) Evaluate on test set
    predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
    preds = predictions.argmax(axis=-1)

    true_labels = [
        [id2label[label_id] for label_id in label_row if label_id != -100]
        for label_row in labels
    ]
    true_preds = [
        [id2label[pred_id] for pred_id, label_id in zip(pred_row, label_row) if label_id != -100]
        for pred_row, label_row in zip(preds, labels)
    ]

    # 13) Classification report
    report_str = classification_report(true_labels, true_preds, digits=4)

    return {
        "true_labels": true_labels,
        "true_preds": true_preds,
        "report": report_str
    }

