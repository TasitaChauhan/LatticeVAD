
import React from 'react';
import { Language } from '../types';
import { SUPPORTED_LANGUAGES } from '../constants';

interface LanguageSelectorProps {
  selected: Language;
  onSelect: (lang: Language) => void;
  disabled?: boolean;
}

export const LanguageSelector: React.FC<LanguageSelectorProps> = ({ selected, onSelect, disabled }) => {
  return (
    <div className="flex flex-col gap-3">
      <label className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em]">
        1. Select Input Language
      </label>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
        {SUPPORTED_LANGUAGES.map((lang) => (
          <button
            key={lang}
            type="button"
            onClick={() => onSelect(lang)}
            disabled={disabled}
            className={`px-3 py-2 rounded-lg text-xs font-mono transition-all border ${
              selected === lang
                ? 'bg-blue-600/20 text-blue-400 border-blue-500 shadow-[0_0_15px_rgba(59,130,246,0.2)]'
                : 'bg-slate-900 text-slate-500 border-slate-800 hover:border-slate-700 hover:text-slate-300'
            } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            {lang.toUpperCase()}
          </button>
        ))}
      </div>
    </div>
  );
};
