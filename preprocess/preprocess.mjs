/**
 * 前処理スクリプト: CSVから埋め込みベクトルを生成してembeddings.jsonに保存する
 * 実行: node preprocess.mjs
 * 事前準備: npm install
 */
import { pipeline, env } from '@xenova/transformers';
import { readFileSync, writeFileSync, existsSync } from 'fs';

const CSV_FILE = 'tetuzuki .csv';
const OUTPUT_FILE = 'embeddings.json';
const MODEL = 'Xenova/multilingual-e5-small';
const BATCH_SIZE = 16;

// --- CSV パーサ (shift_jis 対応、セル内改行対応) ---
function parseCSV(buffer) {
  const text = new TextDecoder('shift-jis').decode(buffer).replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  const records = [];
  let i = 0;
  let record = [];
  let field = '';

  while (i < text.length) {
    if (text[i] === '"') {
      i++;
      while (i < text.length) {
        if (text[i] === '"') {
          if (text[i + 1] === '"') { field += '"'; i += 2; }
          else { i++; break; }
        } else {
          field += text[i++];
        }
      }
    } else if (text[i] === ',') {
      record.push(field); field = ''; i++;
    } else if (text[i] === '\n') {
      record.push(field); field = '';
      records.push(record); record = [];
      i++;
    } else {
      field += text[i++];
    }
  }
  if (field || record.length) { record.push(field); records.push(record); }

  const headers = records[0];
  return records.slice(1)
    .filter(r => r.some(f => f.trim()))
    .map(r => Object.fromEntries(headers.map((h, idx) => [h, (r[idx] ?? '').trim()])));
}

// --- メイン処理 ---
console.log('CSVを読み込み中...');
const rows = parseCSV(readFileSync(CSV_FILE));
console.log(`  ${rows.length} 件`);

// ユースケースをロード (generate_usecases.mjs で生成)
const usecases = existsSync('usecases.json')
  ? JSON.parse(readFileSync('usecases.json', 'utf-8'))
  : {};
if (Object.keys(usecases).length > 0) {
  console.log(`ユースケースを ${Object.keys(usecases).length} 件ロードしました`);
}

// タグをロード (generate_tags.mjs で生成)
const tags = existsSync('tags.json')
  ? JSON.parse(readFileSync('tags.json', 'utf-8'))
  : {};
if (Object.keys(tags).length > 0) {
  console.log(`タグを ${Object.keys(tags).length} 件ロードしました`);
}

const docs = rows.map(r => ({
  title:         r['手続名称']   ?? '',
  officialName:  r['書類正式名称'] ?? '',
  department:    r['担当課']     ?? '',
  section:       r['担当係']     ?? '',
  location:      r['場所']       ?? '',
  purpose:       r['用途']       ?? '',
  notes:         r['留意事項']   ?? '',
  phone:         r['電話番号']   ?? '',
  url:           r['URL']        ?? '',
  online:        r['電子申請']   ?? '',
  tags:          tags[`${r['手続名称'] ?? ''}|${r['担当課'] ?? ''}`] ?? [],
}));

// 埋め込み用テキスト (multilingual-e5 の "passage: " プレフィックス)
const texts = docs.map(d => {
  const uc = usecases[`${d.title}|${d.department}`] ?? [];
  return `passage: ${[d.title, d.department, d.section, d.purpose, d.notes, ...uc].filter(Boolean).join(' ')}`;
});

console.log('\nモデルを読み込み中...');
const extractor = await pipeline('feature-extraction', MODEL, {
  quantized: true,
});

// バッチ処理
const allEmbeddings = [];
for (let i = 0; i < texts.length; i += BATCH_SIZE) {
  const batch = texts.slice(i, i + BATCH_SIZE);
  const output = await extractor(batch, { pooling: 'mean', normalize: true });
  const hiddenSize = output.dims[output.dims.length - 1];
  for (let j = 0; j < batch.length; j++) {
    const start = j * hiddenSize;
    allEmbeddings.push(Array.from(output.data.subarray(start, start + hiddenSize)));
  }
  process.stdout.write(`\r  ${Math.min(i + BATCH_SIZE, texts.length)} / ${texts.length}`);
}
console.log('\n');

writeFileSync(
  OUTPUT_FILE,
  JSON.stringify({ model: MODEL, docs, embeddings: allEmbeddings }, null, 0),
  'utf-8'
);
console.log(`完了: ${OUTPUT_FILE} に ${docs.length} 件の埋め込みを保存しました`);
