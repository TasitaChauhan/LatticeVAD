document.getElementById('testBtn').addEventListener('click', async () => {
  const endpoint = document.getElementById('endpoint').value.trim();
  const apiKey = document.getElementById('apiKey').value.trim();
  const language = document.getElementById('language').value.trim();
  const audioFormat = document.getElementById('audioFormat').value.trim() || 'mp3';
  const audioBase64 = document.getElementById('audioBase64').value.trim();
  const resultEl = document.getElementById('result');

  resultEl.textContent = 'Running test...';

  if (!endpoint || !apiKey || !language || !audioBase64) {
    resultEl.textContent = 'Please fill in endpoint, x-api-key, language, and audioBase64.';
    return;
  }

  const body = {
    language: language,
    audioFormat: audioFormat,
    audioBase64: audioBase64
  };

  try {
    const resp = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey
      },
      body: JSON.stringify(body)
    });

    const text = await resp.text();

    let pretty;
    try { pretty = JSON.stringify(JSON.parse(text), null, 2); } catch { pretty = text; }

    resultEl.textContent = `Status: ${resp.status}\n\n${pretty}`;
  } catch (err) {
    resultEl.textContent = `Request failed: ${err.message || err}`;
  }
});

document.getElementById('clearBtn').addEventListener('click', () => {
  document.getElementById('audioBase64').value = '';
  document.getElementById('result').textContent = '';
});
