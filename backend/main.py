from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
import os
import datetime
from pipeline import MedicalExtractionPipeline  
from utils.postprocessing import save_results_to_gcs

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BUCKET_NAME = "medical-reports-project-build-open-avenues"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../secrets/gcp_key.json"

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)


def timestamped_filename(prefix, ext):
    now = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{now}.{ext}"


@app.post("/upload_text")
async def upload_text(request: Request):
    data = await request.json()
    text = data.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    filename = timestamped_filename("report", "txt")
    gcs_path = f"raw/single_reports/{filename}"

    # Upload to GCS
    blob = bucket.blob(gcs_path)
    blob.upload_from_string(text.encode("utf-8"), content_type="text/plain")

    # Run pipeline directly
    try:
        full_path = f"gs://{BUCKET_NAME}/{gcs_path}"
        pipeline = MedicalExtractionPipeline(input_mode="single_text", input_path=full_path)
        results = pipeline.run()
        save_results_to_gcs(results,gcs_path, BUCKET_NAME)
        return {"message": "✅ Text uploaded and processed", "result": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")


@app.post("/upload_batch")
async def upload_batch(file: UploadFile = File(...)):
    contents = await file.read()
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    filename = timestamped_filename("batch_report", "csv")
    gcs_path = f"raw/batch_reports/{filename}"
    

    # Upload to GCS
    blob = bucket.blob(gcs_path)
    blob.upload_from_string(contents, content_type=file.content_type)

    # Run pipeline directly
    try:
        full_path = f"gs://{BUCKET_NAME}/{gcs_path}"
        pipeline = MedicalExtractionPipeline(input_mode="batch_file", input_path=full_path)
        results = pipeline.run()
        save_results_to_gcs(results, gcs_path, BUCKET_NAME)
        return {"message": "✅ Batch uploaded and processed", "result": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")
