import pandas as pd
from google.cloud import storage
import io

# Initialize a global GCS client once (better than inside function)
gcs_client = storage.Client()

def load_data_with_spark(mode, path):
    """
    mode: 'single_text' or 'batch_file'
    path: local filepath or gs://bucket/object
    """
    if path.startswith("gs://"):
        # Parse bucket and blob name from gs:// URI
        parts = path[5:].split("/", 1)
        bucket_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else ""
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if mode == "single_text":
            # Download text content as string
            content = blob.download_as_text()
            return pd.DataFrame({"ReportText": [content]})

        elif mode == "batch_file":
            # Download blob content as bytes and read CSV from buffer
            content = blob.download_as_bytes()
            return pd.read_csv(io.BytesIO(content))

        else:
            raise ValueError(f"Unknown mode: {mode}")

    else:
        # Local file fallback
        if mode == "single_text":
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            return pd.DataFrame({"ReportText": [text]})

        elif mode == "batch_file":
            return pd.read_csv(path)

        else:
            raise ValueError(f"Unknown mode: {mode}")
