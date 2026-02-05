
import React, { useState } from 'react';

interface Base64InputProps {
  onAnalyze: (base64: string) => void;
  disabled?: boolean;
}

export const Base64Input: React.FC<Base64InputProps> = ({ onAnalyze, disabled }) => {
  const [value, setValue] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (value.trim()) {
      onAnalyze(value.trim());
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full space-y-4">
      <div className="relative">
        <label className="text-xs font-bold text-slate-500 uppercase tracking-[0.2em] mb-2 block">
          Paste Base64 MP3 Code
        </label>
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          disabled={disabled}
          placeholder="FhgTBLEsSxLEsSxLEszEgSBI..."
          className="w-full h-40 bg-slate-950 border border-slate-700 rounded-xl p-4 font-mono text-xs text-blue-400 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none resize-none scrollbar-thin scrollbar-thumb-slate-800"
        />
        <div className="absolute bottom-3 right-3 text-[10px] text-slate-600 font-mono">
          {value.length.toLocaleString()} characters
        </div>
      </div>
      
      <button
        type="submit"
        disabled={disabled || !value.trim()}
        className={`w-full py-4 rounded-xl font-bold uppercase tracking-widest transition-all flex items-center justify-center gap-3 ${
          disabled || !value.trim()
            ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
            : 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-900/20 active:scale-[0.98]'
        }`}
      >
        <i className="fa-solid fa-bolt-lightning"></i>
        Initiate Neural Analysis
      </button>
    </form>
  );
};
