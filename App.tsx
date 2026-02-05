
import React, { useState, useCallback } from 'react';
import { AnalysisState, DetectionRequest, Language } from './types';
import { Base64Input } from './components/Base64Input';
import { AudioUploader } from './components/AudioUploader';
import { ResultDisplay } from './components/ResultDisplay';
import { JsonViewer } from './components/JsonViewer';
import { LanguageSelector } from './components/LanguageSelector';
import { EndpointTester } from './components/EndpointTester';
import { geminiService } from './services/geminiService';

const App: React.FC = () => {
  const [apiKey, setApiKey] = useState('');
  const [selectedLanguage, setSelectedLanguage] = useState<Language>(Language.ENGLISH);
  const [mode, setMode] = useState<'analysis' | 'tester'>('analysis');
  const [analysis, setAnalysis] = useState<AnalysisState>({
    isAnalyzing: false,
    result: null,
    error: null,
    rawRequest: null
  });

  const handleAnalysis = useCallback(async (base64: string) => {
    // 1. Prepare Request with Mandatory Language
    const request: DetectionRequest = {
      language: selectedLanguage,
      audioFormat: 'mp3',
      audioBase64: base64
    };

    setAnalysis({
      isAnalyzing: true,
      result: null,
      error: null,
      rawRequest: request
    });

    try {
      // 2. Perform Analysis
      const response = await geminiService.analyzeVoice(request, apiKey);
      
      if (response.status === 'error') {
        throw new Error(response.message || 'API processing failed');
      }

      setAnalysis(prev => ({
        ...prev,
        isAnalyzing: false,
        result: response,
        error: null
      }));
    } catch (err: any) {
      setAnalysis(prev => ({
        ...prev,
        isAnalyzing: false,
        error: err.message || 'An unexpected error occurred'
      }));
    }
  }, [selectedLanguage]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-blue-600 p-2 rounded-lg shadow-lg shadow-blue-500/20">
              <i className="fa-solid fa-shield-halved text-white text-xl"></i>
            </div>
            <div>
              <h1 className="text-xl font-black tracking-tight text-white uppercase">LatticeVAD</h1>
              <div className="text-[10px] text-slate-400 font-mono tracking-widest uppercase">AI Voice Forensics Engine</div>
            </div>
          </div>

          <div className="hidden md:flex items-center gap-6">
            <div className="flex items-center gap-2 bg-slate-800/50 rounded-lg p-1">
              <button
                onClick={() => setMode('analysis')}
                className={`px-3 py-1 rounded text-xs font-mono uppercase transition-all ${
                  mode === 'analysis'
                    ? 'bg-blue-600/20 text-blue-400'
                    : 'text-slate-400 hover:text-slate-300'
                }`}
              >
                Analysis
              </button>
              <button
                onClick={() => setMode('tester')}
                className={`px-3 py-1 rounded text-xs font-mono uppercase transition-all ${
                  mode === 'tester'
                    ? 'bg-blue-600/20 text-blue-400'
                    : 'text-slate-400 hover:text-slate-300'
                }`}
              >
                Endpoint Tester
              </button>
            </div>
            <div className="h-8 w-px bg-slate-800"></div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></span>
              <span className="text-xs font-mono text-emerald-500 uppercase tracking-tighter">System Online</span>
            </div>
            <div className="h-8 w-px bg-slate-800"></div>
            <div className="flex items-center gap-2">
              <label className="text-[10px] uppercase font-bold text-slate-500">x-api-key</label>
              <input 
                type="password" 
                value={apiKey} 
                onChange={(e) => setApiKey(e.target.value)}
                className="bg-slate-950 border border-slate-700 px-3 py-1 rounded text-xs font-mono text-blue-400 focus:border-blue-500 outline-none w-48"
                placeholder="YOUR_SECRET_API_KEY"
              />
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-12">
        {mode === 'analysis' ? (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Left Column: Input */}
          <div className="lg:col-span-5 space-y-6">
            <section className="bg-slate-900/40 p-8 rounded-3xl border border-slate-800 shadow-2xl space-y-8">
              <div>
                <h2 className="text-xl font-bold text-white mb-2">Analysis Terminal</h2>
                <p className="text-sm text-slate-400">Specify language context and paste the binary stream.</p>
              </div>

              <LanguageSelector 
                selected={selectedLanguage} 
                onSelect={setSelectedLanguage} 
                disabled={analysis.isAnalyzing}
              />
              
              <div className="pt-6 border-t border-slate-800">
                <label className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] mb-4 block">
                  2. Binary Input (Base64)
                </label>
                <div className="space-y-4">
                  <AudioUploader onAnalyze={handleAnalysis} disabled={analysis.isAnalyzing} />
                  <Base64Input 
                    onAnalyze={handleAnalysis}
                    disabled={analysis.isAnalyzing}
                  />
                </div>
              </div>
            </section>

            <section className="bg-slate-800/20 p-6 rounded-2xl border border-slate-800/50">
              <h2 className="text-[10px] font-bold text-slate-500 mb-4 uppercase tracking-[0.2em]">Forensic Logic Rules</h2>
              <ul className="space-y-3 text-xs text-slate-400 font-mono">
                <li className="flex gap-3">
                  <span className="text-blue-500">■</span>
                  <span>LANGUAGE: MANDATORY INPUT context (no auto-detection).</span>
                </li>
                <li className="flex gap-3">
                  <span className="text-blue-500">■</span>
                  <span>ECHO: System reflects provided language in response.</span>
                </li>
                <li className="flex gap-3">
                  <span className="text-blue-500">■</span>
                  <span>VALIDATION: Only TAM, ENG, HIN, MAL, TEL supported.</span>
                </li>
              </ul>
            </section>
          </div>

          {/* Right Column: Results & Debug */}
          <div className="lg:col-span-7 space-y-8">
            
            <section className="bg-slate-900/40 min-h-[400px] p-8 rounded-3xl border border-slate-800 shadow-2xl relative overflow-hidden transition-all">
              {analysis.isAnalyzing ? (
                <div className="h-full flex flex-col items-center justify-center py-12">
                  <div className="relative mb-8">
                    <div className="w-24 h-24 border-4 border-blue-500/20 border-t-blue-500 rounded-full animate-spin"></div>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <i className="fa-solid fa-brain text-blue-400 animate-pulse text-2xl"></i>
                    </div>
                  </div>
                  <div className="text-center">
                    <h3 className="text-xl font-bold text-white mb-2">Neural Forensic Scan</h3>
                    <p className="text-xs text-slate-500 mt-2 font-mono">CONDITIONING_BY_{selectedLanguage.toUpperCase()}...</p>
                  </div>
                </div>
              ) : analysis.error ? (
                <div className="h-full flex flex-col items-center justify-center text-center py-12">
                  <div className="bg-red-500/10 p-6 rounded-full mb-6 border border-red-500/20">
                    <i className="fa-solid fa-triangle-exclamation text-5xl text-red-500"></i>
                  </div>
                  <h3 className="text-2xl font-bold text-white">System Error</h3>
                  <p className="text-red-400 mt-2 font-mono text-sm max-w-sm">{analysis.error}</p>
                </div>
              ) : analysis.result ? (
                <ResultDisplay result={analysis.result} />
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-center opacity-40 py-12">
                  <div className="bg-slate-800 p-8 rounded-full mb-6">
                    <i className="fa-solid fa-satellite-dish text-6xl text-slate-600"></i>
                  </div>
                  <h3 className="text-2xl font-bold text-slate-400">System Ready</h3>
                  <p className="max-w-xs mt-2 mx-auto">Input binary data to begin forensic classification.</p>
                </div>
              )}
            </section>

            <section className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl">
              <h2 className="text-xs font-bold text-slate-500 mb-6 uppercase tracking-[0.2em] flex items-center gap-2">
                <i className="fa-solid fa-terminal text-blue-400"></i>
                REST API Protocol Inspector
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 h-full">
                <JsonViewer 
                  title="Request (language: fixed)" 
                  data={analysis.rawRequest} 
                  type="request"
                />
                <JsonViewer 
                  title="Response (echo: fixed)" 
                  data={analysis.result} 
                  type="response"
                />
              </div>
            </section>
          </div>
        </div>
        ) : (
        <div className="max-w-3xl mx-auto">
          <EndpointTester defaultEndpoint="" />
        </div>
        )}
      </main>

      <footer className="mt-12 py-10 border-t border-slate-900 bg-slate-950/80">
        <div className="max-w-6xl mx-auto px-4 flex flex-col md:flex-row items-center justify-between gap-4 text-center md:text-left">
          <div className="text-[10px] text-slate-500 font-mono">
            SECURE_HANDSHAKE_ACTIVE // CLASSIFIER_v2.5 // PROD_NODE_82
          </div>
          <div className="flex gap-6 text-[10px] text-slate-500 font-bold uppercase tracking-widest">
            <span>ISO/IEC 27001 COMPLIANT</span>
            <a href="#" className="hover:text-blue-400 transition-colors">API Docs</a>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
