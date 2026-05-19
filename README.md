# 練馬区 手続き検索

練馬区が公開するオープンデータをもとに、自然言語でくらしの手続きを検索できる非公式Webアプリです。  
AIによる意味検索をブラウザ内で完結させており、入力した文章はどこにも送信されません。

> **非公式プロジェクトです。** 本アプリは[合同会社Pons](https://www.pons-llc.com/)による試作品であり、練馬区が制作・提供する公式サービスではありません。

---

## 概要

| 項目 | 内容 |
|------|------|
| 検索対象 | 練馬区 手続き情報一覧（2023年9月30日時点） |
| ライセンス | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.ja) に基づきオープンデータを改変して使用 |
| 動作環境 | モダンブラウザ（Chrome / Firefox / Safari / Edge） |
| 通信 | 初回のみモデル（約120MB）をダウンロード、以降はオフライン動作 |

---

## アーキテクチャ

```
preprocess/
  tetuzuki .csv          # 元データ（Shift-JIS）
  generate_usecases.mjs  # Gemini API でユースケース文を生成 → usecases.json
  preprocess.mjs         # 埋め込みベクトルを生成 → embeddings.json

public/
  index.html             # フロントエンド（単一ファイル）
  app.mjs                # ブラウザ内で意味検索を実行
  embeddings.json        # 前処理済みの埋め込みデータ（コミット対象）
```

前処理はローカルで一度だけ実行します。生成された `embeddings.json` を `public/` に配置すれば、あとは静的ファイルの配信だけで動作します。

---

## セットアップ

### 1. 依存パッケージをインストール

```bash
npm install
```

### 2. （任意）ユースケース文を生成

Gemini API でユースケースを生成するとより自然な検索ができます。スキップ可能。

```bash
cd preprocess
GEMINI_API_KEY=your_key node generate_usecases.mjs
```

冪等設計のため、中断した場合は再実行すると続きから処理されます。

### 3. 埋め込みベクトルを生成

```bash
cd preprocess
node preprocess.mjs
```

完了すると `preprocess/embeddings.json` が生成されます。これを `public/embeddings.json` に配置してください。

```bash
cp preprocess/embeddings.json public/embeddings.json
```

### 4. ローカルで確認

```bash
npx serve public
```

---

## 自治体による運用・デプロイ

本アプリは `public/` ディレクトリを静的ファイルとして配信するだけで動作します。  
自治体が自前のデータで運用する場合の手順を以下に示します。

### 1. データを差し替える

`preprocess/` ディレクトリに自治体のCSVファイルを配置し、`preprocess.mjs` の `CSV_FILE` を書き換えてください。

CSVはShift-JIS形式で、以下の列が必要です。

| 列名 | 内容 |
|------|------|
| 手続名称 | 手続きの名称 |
| 書類正式名称 | 書類の正式名称 |
| 担当課 | 担当する課 |
| 担当係 | 担当する係 |
| 場所 | 窓口の場所 |
| 用途 | 手続きの目的・用途 |
| 留意事項 | 注意事項 |
| 電話番号 | 問い合わせ先 |
| URL | 関連ページURL |
| 電子申請 | 電子申請の可否 |

### 2. 前処理を実行する

```bash
npm install
cd preprocess
node preprocess.mjs
cp embeddings.json ../public/embeddings.json
```

### 3. Webサーバーに配置する

`public/` の中身をドキュメントルートに配置するだけで動作します。

**nginx の場合**

```nginx
server {
    listen 80;
    server_name example.lg.jp;
    root /var/www/proc-search/public;
    index index.html;

    location / {
        try_files $uri $uri/ =404;
    }
}
```

**Apache の場合**

```apache
<VirtualHost *:80>
    ServerName example.lg.jp
    DocumentRoot /var/www/proc-search/public
</VirtualHost>
```

**簡易確認（Python）**

```bash
python3 -m http.server 8080 --directory public
```

### 注意事項

- **ライブラリとモデルは事前にダウンロードが必要です。** `app.mjs` はライブラリを jsDelivr CDN から、モデルを Hugging Face からそれぞれ取得します。インターネットに接続できないイントラネット環境では動作しません。完全オフラインで運用する場合は、事前にインターネットに接続できる端末で以下のファイルをダウンロードし、`public/` 内に配置した上でサーバーに展開してください。

  ```bash
  # ライブラリ（約2MB）
  mkdir -p public/vendor
  curl -o public/vendor/transformers.min.js \
    https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js

  # モデルファイル（合計約120MB）
  COMMIT=761b726dd34fb83930e26aab4e9ac3899aa1fa78
  BASE=https://huggingface.co/Xenova/multilingual-e5-small/resolve/$COMMIT
  mkdir -p public/models/Xenova/multilingual-e5-small/onnx
  curl -o public/models/Xenova/multilingual-e5-small/config.json              $BASE/config.json
  curl -o public/models/Xenova/multilingual-e5-small/tokenizer.json           $BASE/tokenizer.json
  curl -o public/models/Xenova/multilingual-e5-small/tokenizer_config.json    $BASE/tokenizer_config.json
  curl -o public/models/Xenova/multilingual-e5-small/special_tokens_map.json  $BASE/special_tokens_map.json
  curl -o public/models/Xenova/multilingual-e5-small/sentencepiece.bpe.model  $BASE/sentencepiece.bpe.model
  curl -o public/models/Xenova/multilingual-e5-small/onnx/model_quantized.onnx $BASE/onnx/model_quantized.onnx
  ```

  ダウンロード後は `app.mjs` の1行目のインポートパスを `./vendor/transformers.min.js` に、`env.localModelPath` を `'./models/'` に変更し、`env.allowRemoteModels` を `false` に設定してください。

- **HTTPS推奨。** 一部のブラウザではHTTPS環境でないとSharedArrayBuffer（WebAssembly高速化）が利用できない場合があります。
- `index.html` の `Content-Security-Policy` ヘッダーは変更しないでください。外部スクリプトの実行を制限するセキュリティ設定です。

---

## 使用ライブラリ・モデル

安全性を確認した上でバージョンを固定して使用しています。

| 用途 | 名称 | バージョン |
|------|------|-----------|
| ブラウザ内推論 | [@xenova/transformers](https://github.com/xenova/transformers.js) | 2.17.2（CDN固定） |
| 意味検索モデル | [Xenova/multilingual-e5-small](https://huggingface.co/Xenova/multilingual-e5-small) | commit `761b726` |
| ユースケース生成（前処理のみ） | Google Gemini API | gemini-3.1-flash-lite |

検索処理はすべてブラウザ内で実行されます。入力テキストは外部に送信されません。

---

## 免責事項

本アプリの利用によって生じたいかなる損害についても、制作者は一切の責任を負いません。  
手続きの詳細は必ず練馬区の公式サイトまたは窓口でご確認ください。
