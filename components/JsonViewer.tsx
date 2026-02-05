
import React from 'react';

interface JsonViewerProps {
  title: string;
  data: any;
  type: 'request' | 'response';
}

export const JsonViewer: React.FC<JsonViewerProps> = ({ title, data, type }) => {
  const formattedData = JSON.stringify(data || { }, (key, value) => {
    // Truncate long base64 strings for display
    if (key === 'audioBase64' && typeof value === 'string' && value.length > 50) {
      return `${value.substring(0, 30)}... [TRUNCATED] ...${value.substring(value.length - 20)}`;
    }
    return value;
  }, 2);

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${type === 'request' ? 'bg-amber-500' : 'bg-blue-500'}`}></span>
          {title}
        </h4>
        <button 
          onClick={async () => {
            try {
              await navigator.clipboard.writeText(JSON.stringify(data || {}, null, 2));
            } catch (err) {
              try {
                // Fallback: create temporary textarea
                const t = document.createElement('textarea');
                t.value = JSON.stringify(data || {}, null, 2);
                document.body.appendChild(t);
                t.select();
                document.execCommand('copy');
                document.body.removeChild(t);
              } catch {
                // eslint-disable-next-line no-alert
                alert('Copy failed: browser blocked clipboard access');
              }
            }
          }}
          className="text-[10px] text-slate-500 hover:text-white transition-colors"
        >
          COPY
        </button>
      </div>
      <div className="flex-1 bg-slate-950 p-4 rounded-lg border border-slate-800 overflow-auto max-h-64 scrollbar-thin scrollbar-thumb-slate-700">
        <pre className="font-mono text-xs text-blue-300 leading-relaxed whitespace-pre-wrap break-all">
          {formattedData || '{}'}
        </pre>
      </div>
    </div>
  );
};
