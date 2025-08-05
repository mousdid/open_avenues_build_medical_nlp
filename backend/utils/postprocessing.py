import io
import pandas as pd
from google.cloud import storage
import os

def convert_numpy_to_python(obj):
    if isinstance(obj, list):
        return [convert_numpy_to_python(o) for o in obj]
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif hasattr(obj, "item"):  # numpy scalar has .item() method
        return obj.item()
    else:
        return obj



# Assume you have a global GCS client initialized:
gcs_client = storage.Client()

def save_results_to_gcs(results, input_path, bucket_name):
    """
    Save extracted results to a CSV file in GCS under 'results/' prefix,
    preserving the filename from input_path.

    Args:
        results (list of dict): Extraction output
        input_path (str): GCS path like 'raw/batch_reports/batch_report_20250805_012043.csv'
        bucket_name (str): Your GCS bucket name
    """

    # Parse filename and directory
    filename = os.path.basename(input_path)

    # Replace 'raw/' with 'results/' prefix
    if input_path.startswith("raw/"):
        output_blob_path = input_path.replace("raw/", "results/", 1)
    else:
        # Just prepend 'results/' folder
        output_blob_path = f"results/{filename}"

    # Convert results list of dict to DataFrame
    df = pd.DataFrame(results)

    # Ensure 4 expected columns present, fill missing with empty strings
    columns = ["ExamName", "findings", "impression", "clinicaldata"]
    for col in columns:
        if col not in df.columns:
            df[col] = ""

    df_to_save = df[columns]

    # Save DataFrame to CSV in-memory buffer
    csv_buffer = io.StringIO()
    df_to_save.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    # Upload to GCS
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(output_blob_path)
    blob.upload_from_string(csv_data, content_type="text/csv")

    print(f"Saved extraction results to gs://{bucket_name}/{output_blob_path}")
    return f"gs://{bucket_name}/{output_blob_path}"
