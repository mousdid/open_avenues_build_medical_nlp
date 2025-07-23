import os
import json
import glob
import random
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import Dataset, DatasetDict, ClassLabel
from seqeval.metrics import classification_report
import re

def run_finetuning(df):
    """
    Fine-tunes a token classification model on the provided DataFrame and returns test predictions and a classification report.
    Args:
        df (pd.DataFrame): DataFrame with columns 'ReportText', 'findings', 'impression', 'clinicaldata', 'ExamName'
    Returns:
        dict: {'true_labels': true_labels, 'true_preds': true_preds, 'report': report_str}
    """


    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        return text

    def tag_tokens(row):
        text = clean_text(row["ReportText"])
        tokens = text.split()
        tags = ["O"] * len(tokens)
        label_spans = []
        for label_name in ["findings", "impression", "clinicaldata", "ExamName"]:
            span = clean_text(row.get(label_name, ""))
            span_tokens = span.split()
            if span_tokens:
                label_spans.append((span_tokens, label_name))
        i = 0
        while i < len(tokens):
            matched = False
            for span_tokens, label in label_spans:
                n = len(span_tokens)
                if tokens[i:i+n] == span_tokens:
                    if n == 1:
                        tags[i] = f"S-{label}"
                    else:
                        tags[i] = f"B-{label}"
                        for j in range(1, n - 1):
                            tags[i+j] = f"I-{label}"
                        tags[i+n-1] = f"E-{label}"
                    i += n
                    matched = True
                    break
            if not matched:
                i += 1
        return tags

    df = df.copy()
    df["labels"] = df.apply(tag_tokens, axis=1)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df)
    })

    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    label_list = sorted({label for row in df["labels"] for label in row})
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    def tokenize_and_align(examples):
        tokenized = tokenizer(examples["ReportText"].split(), truncation=True, is_split_into_words=True)
        labels = []
        word_ids = tokenized.word_ids()
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            else:
                labels.append(label2id[examples["labels"][word_id]])
        tokenized["labels"] = labels
        return tokenized

    tokenized_datasets = dataset_dict.map(tokenize_and_align)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    args = TrainingArguments(
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
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
    preds = predictions.argmax(-1)
    true_labels = [[id2label[label] for label in example if label != -100] for example in labels]
    true_preds = [[id2label[pred] for pred, lab in zip(pred_row, label_row) if lab != -100] for pred_row, label_row in zip(preds, labels)]
    report_str = classification_report(true_labels, true_preds)

    return {'true_labels': true_labels, 'true_preds': true_preds, 'report': report_str}


