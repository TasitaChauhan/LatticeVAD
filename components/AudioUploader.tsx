import React from 'react';

interface AudioUploaderProps {
  onAnalyze: (base64: string) => void;
  disabled?: boolean;
}

export const AudioUploader: React.FC<AudioUploaderProps> = ({ onAnalyze, disabled }) => {
  const handleFile = async (file?: File) => {
    if (!file) return;
    const allowed = ['audio/mpeg', 'audio/mp3', 'audio/wav'];
    if (!allowed.includes(file.type)) {
      try {
        // Fallback: still attempt to read generic audio files
      } catch {
        return;
      }
    }

    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // result is like: data:audio/mp3;base64,AAAA...
      const idx = result.indexOf('base64,');
      const base64 = idx >= 0 ? result.substring(idx + 7) : result;
      onAnalyze(base64);
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className="flex items-center gap-2">
      <label className="w-full">
        <span className="block text-xs font-bold text-slate-500 uppercase tracking-[0.2em] mb-2">Upload Audio File</span>
        <input
          type="file"
          accept="audio/*"
          disabled={disabled}
          onChange={(e) => handleFile(e.target.files?.[0])}
          className="w-full text-sm text-slate-400"
        />
      </label>
    </div>
  );
};

export default AudioUploader;

export {};
