'use client';

import { useState, ChangeEvent } from 'react';

type MedicalPipelineUIProps = {
  onSuccess?: (results: ExtractionResult[], inputTexts: string[]) => void;
};

interface ExtractionResult {
  ExamName: string;
  findings: string;
  impression: string;
  clinicaldata: string;
  confidence: number;
  initialText: string;
}

export default function MedicalPipelineUI({ onSuccess }: MedicalPipelineUIProps) {
  const [singleText, setSingleText] = useState<string>('');
  const [batchFile, setBatchFile] = useState<File | null>(null);
  const [result, setResult] = useState<ExtractionResult[]>([]);
  const [loadingSingle, setLoadingSingle] = useState<boolean>(false);
  const [loadingBatch, setLoadingBatch] = useState<boolean>(false);

  // Run single text extraction
  const runSingleExtraction = async () => {
    if (!singleText.trim()) return;

    setLoadingSingle(true);

    const res = await fetch('http://localhost:8000/upload_text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: singleText }),
    });

    const data = await res.json();
    setResult(data);

    if (onSuccess && data.result) {
      // Adjust as needed based on actual API response structure
      onSuccess(data.result, [singleText]);
    }
    setLoadingSingle(false);
  };

  // Run batch extraction
  const runBatchExtraction = async () => {
    if (!batchFile) return;

    setLoadingBatch(true);

    const formData = new FormData();
    formData.append('file', batchFile);

    const res = await fetch('http://localhost:8000/upload_batch', {
      method: 'POST',
      body: formData,
    });

    const data = await res.json();
    setResult(data);

    if (onSuccess && data.result) {
      // Example: assuming data.result is an array and you have the original texts too
      onSuccess(data.result, []); // You may want to pass the batch texts here if available
    }
    setLoadingBatch(false);
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    setBatchFile(e.target.files?.[0] || null);
  };

  return (
    <div className="max-w-4xl mx-auto mt-12 p-4 space-y-10">
      <h1 className="text-3xl font-bold mb-6 text-center">
        Medical Report Extraction
      </h1>

      {/* Single Text Extraction */}
      <section className="border rounded p-6 shadow">
        <h2 className="text-xl font-semibold mb-4">Single Text Extraction</h2>
        <textarea
          rows={4}
          value={singleText}
          onChange={(e) => setSingleText(e.target.value)}
          placeholder="Paste or type medical report text here..."
          className="w-full p-3 border rounded resize-none"
        />
        <button
          onClick={runSingleExtraction}
          disabled={loadingSingle}
          className="mt-4 bg-blue-600 text-white px-5 py-2 rounded hover:bg-blue-700 disabled:bg-blue-300"
        >
          {loadingSingle ? 'Running...' : 'Run Single Extraction'}
        </button>
      </section>

      {/* Batch CSV Extraction */}
      <section className="border rounded p-6 shadow">
        <h2 className="text-xl font-semibold mb-4">Batch CSV Extraction</h2>
        <input
          type="file"
          accept=".csv"
          onChange={handleFileChange}
          className="border rounded p-2 w-full"
        />
        <button
          onClick={runBatchExtraction}
          disabled={loadingBatch || !batchFile}
          className="mt-4 bg-green-600 text-white px-5 py-2 rounded hover:bg-green-700 disabled:bg-green-300"
        >
          {loadingBatch ? 'Running...' : 'Run Batch Extraction'}
        </button>
      </section>

      {/* Results */}
      {result && (
        <section className="border rounded p-4 bg-gray-900 text-white mt-10 shadow-lg">
          <h2 className="text-2xl font-semibold mb-4">Extraction Results</h2>
          <pre className="whitespace-pre-wrap break-words max-h-96 overflow-auto">
            {JSON.stringify(result, null, 2)}
          </pre>
        </section>
      )}
    </div>
  );
}
