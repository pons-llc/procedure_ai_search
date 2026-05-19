import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js';

// モーダル同意後に初期化を開始する
await new Promise(resolve => {
  document.getElementById('modal-agree').addEventListener('click', () => {
    document.getElementById('modal-overlay').style.display = 'none';
    resolve();
  }, { once: true });
});

const MODEL = 'Xenova/multilingual-e5-small';
const MODEL_REVISION = '761b726dd34fb83930e26aab4e9ac3899aa1fa78';
const TOP_K = 20;

const FILTERS = [
  { id: 'children',      label: '子どもがいる' },
  { id: 'single-parent', label: 'ひとり親'     },
  { id: 'elderly',       label: '高齢・介護'   },
  { id: 'disability',    label: '障害がある'   },
  { id: 'marriage',      label: '結婚'         },
  { id: 'pregnancy',     label: '妊娠'         },
  { id: 'childbirth',    label: '出産'         },
  { id: 'moving',        label: '引っ越し'     },
  { id: 'death',         label: '死亡'         },
];
const activeFilters = new Set();

const queryEl   = document.getElementById('query');
const statusEl  = document.getElementById('status');
const resultsEl = document.getElementById('results');
const filtersEl = document.getElementById('filters');
const progressBar  = document.getElementById('progress-bar');
const progressFill = document.getElementById('progress-fill');

let extractor = null;
let docsData = null;
let embMatrix = null;

// --- 埋め込みデータ読み込み ---
setStatus('embeddings.json を読み込み中...');
try {
  const res = await fetch('embeddings.json');
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const json = await res.json();
  docsData  = json.docs;
  embMatrix = json.embeddings.map(e => new Float32Array(e));
} catch (e) {
  setStatus('embeddings.json の読み込みに失敗しました。前処理を実行してください。', 'error');
  throw e;
}

// --- モデル読み込み ---
setStatus('AIモデルを読み込み中... (初回はダウンロードに時間がかかります)');
progressBar.style.display = 'block';

env.allowRemoteModels = true;

extractor = await pipeline('feature-extraction', MODEL, {
  quantized: true,
  revision: MODEL_REVISION,
  progress_callback: (p) => {
    if (p.status === 'progress' && p.total) {
      const pct = Math.round(p.loaded / p.total * 100);
      progressFill.style.width = pct + '%';
      setStatus(`モデルを読み込み中: ${p.file} (${pct}%)`);
    } else if (p.status === 'done') {
      progressFill.style.width = '100%';
    }
  },
});

progressBar.style.display = 'none';
queryEl.disabled = false;
queryEl.focus();
setStatus(`準備完了 — ${docsData.length} 件の手続きを検索できます`, 'ready');

// --- フィルターチップ描画 ---
FILTERS.forEach(f => {
  const chip = document.createElement('button');
  chip.className = 'chip';
  chip.textContent = f.label;
  chip.addEventListener('click', () => {
    if (activeFilters.has(f.id)) {
      activeFilters.delete(f.id);
      chip.classList.remove('active');
    } else {
      activeFilters.add(f.id);
      chip.classList.add('active');
    }
    const q = queryEl.value.trim();
    if (q) search(q); else resultsEl.replaceChildren();
  });
  filtersEl.appendChild(chip);
});

// --- 検索 ---
let debounceTimer = null;
queryEl.addEventListener('input', () => {
  clearTimeout(debounceTimer);
  const q = queryEl.value.trim();
  if (!q) { resultsEl.replaceChildren(); return; }
  debounceTimer = setTimeout(() => search(q), 300);
});

function getFilteredIndices() {
  if (activeFilters.size === 0) return null;
  return docsData.reduce((acc, d, i) => {
    const match = [...activeFilters].every(id => (d.tags ?? []).includes(id));
    if (match) acc.push(i);
    return acc;
  }, []);
}

async function search(queryText) {
  setStatus('検索中...', '');
  const output = await extractor(`query: ${queryText}`, { pooling: 'mean', normalize: true });
  const qVec = new Float32Array(output.data);

  const indices = getFilteredIndices() ?? embMatrix.map((_, i) => i);
  const scored = indices.map(i => ({ i, score: cosine(qVec, embMatrix[i]) }));
  scored.sort((a, b) => b.score - a.score);

  renderResults(scored.slice(0, TOP_K));
  setStatus(`準備完了 — ${docsData.length} 件の手続きを検索できます`, 'ready');
}

function cosine(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text) e.textContent = text;
  return e;
}

function renderResults(scored) {
  resultsEl.replaceChildren();

  if (!scored.length) {
    resultsEl.appendChild(el('div', 'empty', '結果が見つかりませんでした'));
    return;
  }

  for (const { i } of scored) {
    const d = docsData[i];
    const card = el('div', 'result-card');

    const titleRow = el('div', 'card-title', d.title);
    if (d.online && d.online !== '×') {
      titleRow.appendChild(el('span', 'badge', '電子申請可'));
    }
    card.appendChild(titleRow);

    if (d.purpose) card.appendChild(el('div', 'card-purpose', d.purpose));

    const meta = el('div', 'card-meta');
    if (d.department) {
      meta.appendChild(el('span', null, `🏢 ${d.department}${d.section ? ' ' + d.section : ''}`));
    }
    if (d.location) meta.appendChild(el('span', null, `📍 ${d.location}`));
    if (d.phone)    meta.appendChild(el('span', null, `📞 ${d.phone}`));
    card.appendChild(meta);

    if (d.notes) card.appendChild(el('div', 'card-notes', `⚠️ ${d.notes}`));

    const urls = (d.url || '').split(/[,\s]+/).filter(u => u.startsWith('http'));
    if (urls.length) {
      const links = el('div', 'card-links');
      for (const u of urls) {
        const a = document.createElement('a');
        a.href = u;
        a.textContent = '🔗 詳細ページ';
        a.target = '_blank';
        a.rel = 'noopener noreferrer';
        links.appendChild(a);
      }
      card.appendChild(links);
    }

    resultsEl.appendChild(card);
  }
}

function setStatus(msg, cls = '') {
  statusEl.textContent = msg;
  statusEl.className = cls;
}
