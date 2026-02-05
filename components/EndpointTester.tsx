import React, { useState } from 'react';

interface EndpointTesterProps {
  defaultEndpoint?: string;
}

export const EndpointTester: React.FC<EndpointTesterProps> = ({ defaultEndpoint = '' }) => {
  const [endpoint, setEndpoint] = useState(defaultEndpoint);
  const [apiKey, setApiKey] = useState('');
  const [language, setLanguage] = useState('English');
  const [audioFormat, setAudioFormat] = useState('mp3');
  const [audioBase64, setAudioBase64] = useState('');
  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleTest = async () => {
    if (!endpoint.trim() || !apiKey.trim() || !language.trim() || !audioBase64.trim()) {
      setResult('⚠️ Please fill in all required fields: endpoint, x-api-key, language, and audioBase64.');
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const body = {
        language: language.trim(),
        audioFormat: audioFormat.trim() || 'mp3',
        audioBase64: audioBase64.trim()
      };

      const resp = await fetch(endpoint.trim(), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': apiKey.trim()
        },
        body: JSON.stringify(body)
      });

      const text = await resp.text();
      let pretty: string;
      try {
        pretty = JSON.stringify(JSON.parse(text), null, 2);
      } catch {
        pretty = text;
      }

      setResult(`HTTP ${resp.status}\n\n${pretty}`);
    } catch (err: any) {
      setResult(`❌ Request failed: ${err?.message || String(err)}`);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setAudioBase64('');
    setResult(null);
  };

  return (
    <div className="bg-slate-900/40 p-8 rounded-3xl border border-slate-800 shadow-2xl space-y-6">
      <div>
        <h2 className="text-xl font-bold text-white mb-2">Endpoint Tester</h2>
        <p className="text-sm text-slate-400">Test your deployed API by sending a request to your endpoint.</p>
      </div>

      <div className="space-y-4">
        <div>
          <label className="text-xs font-bold text-slate-500 uppercase tracking-[0.2em] mb-2 block">
            x-api-key *
          </label>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="Enter x-api-key"
            disabled={loading}
            className="w-full bg-slate-950 border border-slate-700 rounded-xl px-4 py-2 text-xs font-mono text-blue-400 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
          />
        </div>

        <div>
          <label className="text-xs font-bold text-slate-500 uppercase tracking-[0.2em] mb-2 block">
            Endpoint URL *
          </label>
          <input
            type="text"
            value={endpoint}
            onChange={(e) => setEndpoint(e.target.value)}
            placeholder="https://your-service.run.app/api/voice-detection"
            disabled={loading}
            className="w-full bg-slate-950 border border-slate-700 rounded-xl px-4 py-2 text-xs font-mono text-blue-400 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
          />
        </div>

        <div>
          <label className="text-xs font-bold text-slate-500 uppercase tracking-[0.2em] mb-2 block">
            Language *
          </label>
          <input
            type="text"
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            placeholder="English"
            disabled={loading}
            className="w-full bg-slate-950 border border-slate-700 rounded-xl px-4 py-2 text-xs font-mono text-blue-400 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
          />
        </div>

        <div>
          <label className="text-xs font-bold text-slate-500 uppercase tracking-[0.2em] mb-2 block">
            Audio Format *
          </label>
          <input
            type="text"
            value={audioFormat}
            onChange={(e) => setAudioFormat(e.target.value)}
            placeholder="mp3"
            disabled={loading}
            className="w-full bg-slate-950 border border-slate-700 rounded-xl px-4 py-2 text-xs font-mono text-blue-400 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
          />
        </div>

        <div>
          <label className="text-xs font-bold text-slate-500 uppercase tracking-[0.2em] mb-2 block">
            Audio Base64 *
          </label>
          <textarea
            value={audioBase64}
            onChange={(e) => setAudioBase64(e.target.value)}
            placeholder="Paste base64 encoded audio here..."
            disabled={loading}
            className="w-full h-40 bg-slate-950 border border-slate-700 rounded-xl px-4 py-2 text-xs font-mono text-blue-400 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none resize-none"
          />
        </div>

        <div className="flex gap-3">
          <button
            onClick={handleTest}
            disabled={loading}
            className={`flex-1 py-3 rounded-xl font-bold uppercase tracking-widest transition-all flex items-center justify-center gap-2 ${
              loading
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-900/20 active:scale-[0.98]'
            }`}
          >
            <i className="fa-solid fa-paper-plane"></i>
            {loading ? 'Testing...' : 'Test Endpoint'}
          </button>
          <button
            onClick={handleClear}
            disabled={loading}
            className="px-6 py-3 rounded-xl font-bold uppercase tracking-widest bg-slate-800 text-slate-300 hover:bg-slate-700 transition-all"
          >
            Clear
          </button>
        </div>
      </div>

      {result && (
        <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-700 overflow-auto max-h-64">
          <pre className="font-mono text-xs text-green-300 whitespace-pre-wrap break-all leading-relaxed">
            {result}
          </pre>
        </div>
      )}
    </div>
  );
};
