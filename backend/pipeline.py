
from utils.preprocessing import preprocess_data
from utils.extraction import run_extraction
from utils.data_loader import load_data_with_spark
from utils.postprocessing import convert_numpy_to_python

class MedicalExtractionPipeline:
    def __init__(self, input_mode="single_text", input_path=None):
        self.input_mode = input_mode
        self.input_path = input_path

    def run(self):
        # Step 1: Load data
        try:
            batch_raw_texts = load_data_with_spark(self.input_mode, self.input_path)
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

        # Step 2: Preprocess
        cleaned_batch = preprocess_data(batch_raw_texts)

        # Step 3: Extract
        raw_results = run_extraction(cleaned_batch)

        # Step 4: Postprocess
        results = convert_numpy_to_python(raw_results)
        

        return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python pipeline.py <input_mode> <input_path>")
        print("Example: python pipeline.py single_text data/reports/report1.txt")
        sys.exit(1)

    input_mode = sys.argv[1]
    input_path = sys.argv[2]

    pipeline = MedicalExtractionPipeline(input_mode, input_path)
    final_results = pipeline.run()

