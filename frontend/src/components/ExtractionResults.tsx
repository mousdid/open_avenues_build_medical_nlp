'use client';

import React from 'react';

type ExtractionResult = {
  ExamName: string;
  findings: string;
  impression: string;
  clinicaldata: string;
  confidence: number;
  initialText: string;
};

type ResultsProps = {
  results: ExtractionResult[]; // Array for batch or single item array
  onBack: () => void;
};

export default function ExtractionResults({ results, onBack }: ResultsProps) {
  return (
    <div className="max-w-4xl mx-auto p-6 relative">
      <button
        onClick={onBack}
        className="absolute top-6 right-6 bg-gray-300 px-4 py-2 rounded hover:bg-gray-400"
      >
        ← Back
      </button>

      <h1 className="text-3xl font-bold mb-6 text-center">Extraction Results</h1>

      {results.length === 0 && <p>No results to display.</p>}

      <div className="space-y-8 max-h-[70vh] overflow-y-auto">
        {results.map((res, i) => (
          <div
            key={i}
            className="border p-4 rounded shadow bg-white text-black"
          >
            <h2 className="text-xl font-semibold mb-2">
              Report {results.length > 1 ? i + 1 : ""}
            </h2>

            <div className="mb-4 p-2 bg-gray-100 rounded whitespace-pre-wrap font-mono text-sm">
              <strong>Original Text:</strong>
              <br />
              {res.initialText}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h3 className="font-semibold">Exam Name</h3>
                <p>{res.ExamName || "—"}</p>
              </div>
              <div>
                <h3 className="font-semibold">Findings</h3>
                <p>{res.findings || "—"}</p>
              </div>
              <div>
                <h3 className="font-semibold">Impression</h3>
                <p>{res.impression || "—"}</p>
              </div>
              <div>
                <h3 className="font-semibold">Clinical Data</h3>
                <p>{res.clinicaldata || "—"}</p>
              </div>
            </div>

            <div className="mt-4 text-sm text-gray-600">
              Confidence: {(res.confidence * 100).toFixed(2)}%
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
