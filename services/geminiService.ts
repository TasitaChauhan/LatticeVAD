
import { GoogleGenAI, Type } from "@google/genai";
import { DetectionRequest, DetectionResponse, Language, Classification } from "../types";
import { SYSTEM_INSTRUCTION } from "../constants";

export class GeminiService {
  // analyzeVoice accepts an optional apiKey so the UI can provide it at runtime.
  async analyzeVoice(request: DetectionRequest, apiKey?: string): Promise<DetectionResponse> {
    try {
      const key = apiKey || process.env.API_KEY || '';
      if (!key) {
        return {
          status: 'error',
          language: request.language,
          classification: Classification.HUMAN,
          confidenceScore: 0,
          explanation: 'Missing API key',
          message: 'No API key provided to geminiService'
        };
      }

      const ai = new GoogleGenAI({ apiKey: key });

      const response = await ai.models.generateContent({
        model: 'gemini-3-flash-preview',
        contents: [
          {
            parts: [
              {
                inlineData: {
                  mimeType: 'audio/mp3',
                  data: request.audioBase64
                }
              },
              {
                text: `Classify this audio sample. The known spoken language is: ${request.language}. Use this as context only. Do NOT perform language detection. Return the same language in the response.`
              }
            ]
          }
        ],
        config: {
          systemInstruction: SYSTEM_INSTRUCTION,
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              status: { type: Type.STRING },
              language: { type: Type.STRING },
              classification: { type: Type.STRING },
              confidenceScore: { type: Type.NUMBER },
              explanation: { type: Type.STRING }
            },
            required: ["status", "language", "classification", "confidenceScore", "explanation"]
          }
        }
      });

      const resultText = (response as any).text || '';
      let parsed: any = null;
      try {
        parsed = JSON.parse(resultText);
      } catch (err) {
        return {
          status: 'error',
          language: request.language,
          classification: Classification.HUMAN,
          confidenceScore: 0,
          explanation: 'Failed to parse model response as JSON',
          message: (err as Error).message
        };
      }

      // Ensure required fields exist and coerce where appropriate
      const status = parsed.status === 'success' ? 'success' : 'error';
      const classification = parsed.classification === Classification.AI_GENERATED ? Classification.AI_GENERATED : Classification.HUMAN;
      const confidence = typeof parsed.confidenceScore === 'number' ? parsed.confidenceScore : 0;
      const explanation = typeof parsed.explanation === 'string' ? parsed.explanation : String(parsed.explanation || '');

      return {
        status,
        language: request.language, // enforce echo
        classification,
        confidenceScore: confidence,
        explanation
      };
    } catch (error: any) {
      console.error("Gemini Analysis Error:", error);
      return {
        status: 'error',
        language: request.language,
        classification: Classification.HUMAN,
        confidenceScore: 0,
        explanation: "Analysis failed",
        message: error?.message || "Unknown error during processing"
      } as DetectionResponse;
    }
  }
}

export const geminiService = new GeminiService();
