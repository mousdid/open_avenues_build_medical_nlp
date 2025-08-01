import pandas as pd
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re
from tqdm import tqdm
import pandas as pd
import json
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def extract_metadata_with_labels(text):
    exam = re.search(r'EXAM:\s*(.*?)(?:\s*CLINICAL|FINDINGS:|IMPRESSION:|$)', text, re.IGNORECASE)
    clinical = re.search(r'CLINICAL HISTORY:\s*(.*?)(?:COMPARISON:|TECHNIQUE:|FINDINGS:|IMPRESSION:|$)', text, re.IGNORECASE)
    findings = re.search(r'FINDINGS:\s*(.*?)(?:IMPRESSION:|$)', text, re.IGNORECASE)
    impression = re.search(r'IMPRESSION:\s*(.*?)(?:$)', text, re.IGNORECASE)
    return {
        'ExamName': f"EXAM: {exam.group(1).strip()}" if exam else "",
        'clinicaldata': f"CLINICAL HISTORY: {clinical.group(1).strip()}" if clinical else "",
        'findings': f"FINDINGS: {findings.group(1).strip()}" if findings else "",
        'impression': f"IMPRESSION: {impression.group(1).strip()}" if impression else ""
    }

def format_few_shot(row):
    return f"""Report:\n{row['ReportText']}\n\nOutput:\n{{
  "ExamName": "{row['ExamName']}",
  "clinicaldata": "{row['clinicaldata']}",
  "findings": "{row['findings']}",
  "impression": "{row['impression']}"
}}\n"""

def build_few_shots_prompt(few_shot_df):
    return "\n---\n".join(few_shot_df.apply(format_few_shot, axis=1))

def build_prompt(report_text, few_shots_prompt):
    return f"""You are a clinical language processing expert working in English. All your output must be in English. You will return the extracted results in valid JSON using English field labels and English content only. Your task is to extract structured information from unstructured radiology reports.

Each report typically contains the following sections:
- EXAM or Exam
- EXAM DATE
- CLINICAL HISTORY or INDICATION
- COMPARISON
- TECHNIQUE
- FINDINGS
- IMPRESSION

You will extract the following fields:
- "ExamName": contains the exam name and optionally technique or comparison.
- "clinicaldata": the clinical history, indication, or reported symptoms.
- "findings": the radiologist's findings from the scan.
- "impression": the final interpretation or conclusion.

Step-by-step approach:
1. Find and group information starting from **EXAM**. Include **EXAM DATE**, **TECHNIQUE**, and **COMPARISON** if found in the same section.
2. Extract **CLINICAL HISTORY**, **INDICATION**, or **History** for "clinicaldata".
3. Extract the **FINDINGS** section exactly.
4. Extract the **IMPRESSION** section exactly.

Return the results in strict JSON format like this:
{{
  "ExamName": "...",
  "clinicaldata": "...",
  "findings": "...",
  "impression": "..."
}}

If any section is not available, return an empty string for that field.

Here are a few examples:

{few_shots_prompt}
---
Now extract the following report:

Report:
{report_text}

Output:
"""

def build_chat_messages(few_shot_df, test_report):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a clinical language processing expert. Your task is to extract structured fields from radiology reports. "
                "You will be given several examples, followed by a new report. "
                "Your response must match the format of the previous examples exactly, using valid JSON with the following fields only: "
                '"ExamName", "clinicaldata", "findings", "impression".\n'
                "• Use exact phrasing from the report.\n"
                "• Include section labels if they appear in the text.\n"
                "• If a section is missing, return an empty string for it.\n"
                "• All output must be in English. Do not generate Chinese or non-English characters."
            )
        }
    ]
    for _, row in few_shot_df.iterrows():
        messages.append({
            "role": "user",
            "content": f"Report:\n{row['ReportText']}\n\nOutput (JSON):"
        })
        messages.append({
            "role": "assistant",
            "content": json.dumps({
                "ExamName": row["ExamName"],
                "clinicaldata": row["clinicaldata"],
                "findings": row["findings"],
                "impression": row["impression"]
            }, indent=2)
        })
    messages.append({
        "role": "user",
        "content": f"Report:\n{test_report}\n\nOutput (JSON):"
    })
    return messages

def run_extraction_qwen(test_index, df, few_shot_df, tokenizer, model):
    messages = build_chat_messages(few_shot_df, df.loc[test_index, 'ReportText'])
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cpu")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=512,
            temperature=0.1
        )
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return re.sub(r'[^\x00-\x7F]+', '', decoded)

def batch_extraction(df, few_shot_df, tokenizer, model, start_index, end_index):
    results = []
    for i in tqdm(range(start_index, end_index), desc="Running extractions"):
        output_str = run_extraction_qwen(i, df, few_shot_df, tokenizer, model)
        try:
            parsed = json.loads(output_str)
        except json.JSONDecodeError:
            parsed = {
                "ExamName": "",
                "clinicaldata": "",
                "findings": "",
                "impression": ""
            }
        results.append(parsed)
    return pd.DataFrame(results)
