'use client';

import { useState } from 'react';
import MedicalPipelineUI from '@/components/MedicalPipelineUI';
import ExtractionResults, { ExtractionResult } from '@/components/ExtractionResults';

export default function MedicalApp() {
  const [results, setResults] = useState<ExtractionResult[] | null>(null);
  const [originalTexts, setOriginalTexts] = useState<string[]>([]);

  // This will be called by MedicalPipelineUI after successful extraction
  // We'll pass in results from API and the original input texts for display
  const handleSuccess = (extractionResults: ExtractionResult[], inputTexts: string[]) => {
    setResults(extractionResults);
    setOriginalTexts(inputTexts);
  };

  const handleBack = () => {
    setResults(null);
    setOriginalTexts([]);
  };

  return (
    <>
      {!results ? (
        <MedicalPipelineUI onSuccess={handleSuccess} />
      ) : (
        <ExtractionResults
          results={results.map((res, i) => ({
            ...res,
            initialText: originalTexts[i] || "",
          }))}
          onBack={handleBack}
        />
      )}
    </>
  );
}
