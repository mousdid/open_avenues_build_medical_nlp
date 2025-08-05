import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import os

def run_extraction(cleaned_batch):
    """
    cleaned_batch: pandas DataFrame with a 'ReportText' column of strings.
    Returns list of dicts, each dict containing extracted entities and confidence.
    """

    # Load Hugging Face token and model repo name
    hf_token_path = "../secrets/hugging_face.txt"
    with open(hf_token_path, "r") as f:
        hf_token = f.read().strip()

    model_name_or_path = "mo191919/ner_model"

    # Load tokenizer and model from Hugging Face Hub with auth token
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=hf_token)
    model = AutoModelForTokenClassification.from_pretrained(model_name_or_path, token=hf_token)

    model.eval()

    # Use pipeline for convenience (NER pipeline handles postprocessing)
    nlp = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",  # to aggregate tokens into entities
        device=0 if torch.cuda.is_available() else -1
    )

    results = []
    for text in cleaned_batch["ReportText"]:
        ner_results = nlp(text)

        # Aggregate extracted entities by label
        extracted = {}
        for entity in ner_results:
            label = entity["entity_group"]
            val = entity["word"]
            conf = entity["score"]
            if label not in extracted:
                extracted[label] = {"text": val, "confidence": conf}
            else:
                # Optionally concatenate entities of the same label
                extracted[label]["text"] += " " + val
                # For confidence, you could keep max or average, here max:
                extracted[label]["confidence"] = max(extracted[label]["confidence"], conf)

        # Flatten dict if you want only text or confidence separately
        flat_result = {f"{k}": v["text"] for k, v in extracted.items()}
        flat_result["confidence"] = max(v["confidence"] for v in extracted.values()) if extracted else 0.0

        results.append(flat_result)

    return results
