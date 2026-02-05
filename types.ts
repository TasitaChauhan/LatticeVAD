
export enum Language {
  TAMIL = 'Tamil',
  ENGLISH = 'English',
  HINDI = 'Hindi',
  MALAYALAM = 'Malayalam',
  TELUGU = 'Telugu'
}

export enum Classification {
  AI_GENERATED = 'AI_GENERATED',
  HUMAN = 'HUMAN'
}

export interface DetectionRequest {
  language: Language;
  audioFormat: 'mp3';
  audioBase64: string;
}

export interface DetectionResponse {
  status: 'success' | 'error';
  language: Language;
  classification: Classification;
  confidenceScore: number;
  explanation: string;
  message?: string;
}

export interface AnalysisState {
  isAnalyzing: boolean;
  result: DetectionResponse | null;
  error: string | null;
  rawRequest: DetectionRequest | null;
}
