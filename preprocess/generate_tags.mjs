/**
 * タグ付けスクリプト: Gemini APIで各手続きに対象者タグを付与し tags.json に保存する
 * 実行: GEMINI_API_KEY=... node generate_tags.mjs
 * 冪等: 既存エントリはスキップ。中断後の再実行で続きから処理される。
 */
import { GoogleGenerativeAI } from '@google/generative-ai';
import { readFileSync, writeFileSync, existsSync } from 'fs';

const CSV_FILE    = 'tetuzuki .csv';
const OUTPUT_FILE = 'tags.json';
const MODEL       = 'gemini-3.1-flash-lite';
const BATCH_SIZE  = 20;
const RETRY_MAX   = 3;

const VALID_TAGS = [
  'children', 'single-parent', 'elderly', 'disability',
  'marriage', 'pregnancy', 'childbirth', 'moving', 'death',
];

const TAG_DESCRIPTIONS = `
【対象者】
- "children"      : 子どもがいる（子育て・育児・就学・出産・保育・乳幼児など、子ども全般に関わる手続き）
- "single-parent" : ひとり親（母子・父子家庭を対象とした手続き）
- "elderly"       : 高齢・介護（高齢者本人、または介護する家族向けの手続き）
- "disability"    : 障害がある（身体・知的・精神障害のある方、またはその家族向けの手続き）

【ライフイベント】
- "marriage"      : 結婚（婚姻届・氏名変更・住所変更など結婚に伴う手続き）
- "pregnancy"     : 妊娠（妊娠届・母子手帳・妊婦健診など妊娠中の手続き）
- "childbirth"    : 出産（出生届・児童手当・乳幼児医療など出産後の手続き）
- "moving"        : 引っ越し（転入・転出・転居届など住所変更に伴う手続き）
- "death"         : 死亡（死亡届・相続・年金停止など死亡に伴う手続き）
`.trim();

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

function buildBatchPrompt(rows) {
  const items = rows.map((r, i) =>
    `${i + 1}. 【手続名称】${r.title} 【用途】${r.purpose} 【担当課】${r.department}${r.notes ? ` 【留意事項】${r.notes}` : ''}`
  ).join('\n');
  return `あなたは練馬区の行政手続きを分類するアシスタントです。
以下の${rows.length}件の手続きについて、それぞれ該当する対象者タグを選んでください。

タグの定義:
${TAG_DESCRIPTIONS}

ルール:
- 複数該当する場合はすべて選ぶ
- 該当なしの場合は空配列
- タグIDのみを返す（日本語不要）

${items}

出力形式: ${rows.length}要素のJSON配列（各要素はタグID文字列の配列）。説明文・マークダウン・コードブロック不要。
例(3件の場合): [["children"], ["elderly", "disability"], []]`;
}

async function generateBatch(model, rows, retries = 0) {
  try {
    const result = await model.generateContent(buildBatchPrompt(rows));
    const text = result.response.text().trim()
      .replace(/^```(?:json)?\n?/, '').replace(/\n?```$/, '');
    const parsed = JSON.parse(text);
    if (!Array.isArray(parsed) || parsed.length !== rows.length) throw new Error('length mismatch');
    return parsed.map(tags =>
      Array.isArray(tags) ? tags.filter(t => VALID_TAGS.includes(t)) : []
    );
  } catch (err) {
    if (retries < RETRY_MAX) {
      await new Promise(r => setTimeout(r, 1000 * 2 ** retries));
      return generateBatch(model, rows, retries + 1);
    }
    console.error(`\n  [WARN] バッチ失敗、空配列で保存 (${err.message})`);
    return rows.map(() => []);
  }
}

// --- メイン処理 ---
const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  console.error('エラー: 環境変数 GEMINI_API_KEY が設定されていません');
  process.exit(1);
}

const rows = parseCSV(readFileSync(CSV_FILE));
const docs = rows.map(r => ({
  title:      r['手続名称'] ?? '',
  purpose:    r['用途']     ?? '',
  department: r['担当課']   ?? '',
  notes:      r['留意事項'] ?? '',
}));

const existing = existsSync(OUTPUT_FILE)
  ? JSON.parse(readFileSync(OUTPUT_FILE, 'utf-8'))
  : {};

const seenKeys = new Set(Object.keys(existing));
const pending = [];
for (const d of docs) {
  const key = `${d.title}|${d.department}`;
  if (!seenKeys.has(key)) {
    seenKeys.add(key);
    pending.push(d);
  }
}
console.log(`処理済み: ${Object.keys(existing).length} 件 / 未処理: ${pending.length} 件`);

if (pending.length === 0) {
  console.log('すべて処理済みです。');
  process.exit(0);
}

const genAI = new GoogleGenerativeAI(apiKey);
const model = genAI.getGenerativeModel({ model: MODEL });

const results = { ...existing };
for (let i = 0; i < pending.length; i += BATCH_SIZE) {
  const chunk = pending.slice(i, i + BATCH_SIZE);
  const tags = await generateBatch(model, chunk);
  chunk.forEach((d, idx) => {
    results[`${d.title}|${d.department}`] = tags[idx];
  });
  writeFileSync(OUTPUT_FILE, JSON.stringify(results, null, 2), 'utf-8');
  process.stdout.write(`\r  ${Math.min(i + BATCH_SIZE, pending.length)} / ${pending.length}`);
}
console.log(`\n完了: ${OUTPUT_FILE} に ${Object.keys(results).length} 件を保存しました`);
