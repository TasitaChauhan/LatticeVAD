
import React from 'react';
import { DetectionResponse, Classification } from '../types';

interface ResultDisplayProps {
  result: DetectionResponse;
}

export const ResultDisplay: React.FC<ResultDisplayProps> = ({ result }) => {
  const isAI = result.classification === Classification.AI_GENERATED;
  const rawScore = Number(result.confidenceScore) || 0;
  const scorePercent = Math.max(0, Math.min(100, Math.round(rawScore * 100)));

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-bold text-white flex items-center gap-2">
          <i className="fa-solid fa-microscope text-blue-400"></i>
          Forensic Report
        </h3>
        <span className="px-3 py-1 rounded-full bg-slate-800 text-xs font-mono text-slate-400 border border-slate-700">
          SECURE_SCAN_v2.1
        </span>
      </div>

      <div className={`p-6 rounded-2xl border-2 shadow-2xl ${
        isAI 
          ? 'bg-red-950/20 border-red-500/50 shadow-red-900/20' 
          : 'bg-emerald-950/20 border-emerald-500/50 shadow-emerald-900/20'
      }`}>
        <div className="flex flex-col md:flex-row md:items-center gap-6">
          <div className="flex-1">
            <div className="text-sm uppercase tracking-widest text-slate-400 mb-1 font-semibold">
              Classification
            </div>
            <div className={`text-4xl font-black mb-2 flex items-center gap-3 ${
              isAI ? 'text-red-400' : 'text-emerald-400'
            }`}>
              {isAI ? (
                <i className="fa-solid fa-robot"></i>
              ) : (
                <i className="fa-solid fa-user-check"></i>
              )}
              {result.classification}
            </div>
            <div className="text-slate-300 leading-relaxed italic">
              &quot;{result.explanation}&quot;
            </div>
          </div>

          <div className="flex flex-col items-center justify-center p-4 bg-slate-900/50 rounded-xl min-w-[140px] border border-white/5">
            <div className="text-xs uppercase tracking-tighter text-slate-500 mb-1">Confidence</div>
            <div className={`text-3xl font-mono font-bold ${
              isAI ? 'text-red-500' : 'text-emerald-500'
            }`}>
              {scorePercent}%
            </div>
            <div className="w-full bg-slate-800 h-1.5 rounded-full mt-2 overflow-hidden">
              <div 
                className={`h-full rounded-full transition-all duration-1000 ${
                  isAI ? 'bg-red-500' : 'bg-emerald-500'
                }`}
                style={{ width: `${scorePercent}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-slate-800/40 p-4 rounded-xl border border-slate-700">
          <div className="text-xs text-slate-500 uppercase mb-2">Detected Language</div>
          <div className="text-lg font-semibold text-white flex items-center gap-2">
            <i className="fa-solid fa-language text-blue-400"></i>
            {result.language}
          </div>
        </div>
        <div className="bg-slate-800/40 p-4 rounded-xl border border-slate-700">
          <div className="text-xs text-slate-500 uppercase mb-2">Analysis Status</div>
          <div className={`text-lg font-semibold flex items-center gap-2 ${result.status === 'success' ? 'text-emerald-400' : 'text-red-400'}`}>
            <i className={`fa-solid ${result.status === 'success' ? 'fa-circle-check' : 'fa-triangle-exclamation'}`}></i>
            {String(result.status).toUpperCase()}
          </div>
          {result.message ? (
            <div className="text-xs text-red-300 mt-2 font-mono">{result.message}</div>
          ) : null}
        </div>
      </div>
    </div>
  );
};
