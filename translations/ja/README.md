# translations/ja — 実験データの日本語訳

このフォルダは、本リポジトリの実験で使った**中国語データの日本語訳**です。
データは CharacterEval（中国語ロールプレイ・ベンチマーク）由来のため、人格スキーマ・
対話文脈・モデル生成出力がすべて中国語です。中身を確認したい読者向けに、全データを
日本語へ翻訳しました（翻訳: Claude）。読みたい人だけ参照してください。

## 中身

| パス | 内容 |
|---|---|
| `facets/<name>.json` | 77体の Narrative Schema（人格スキーマ）。元の `../../facets/` と同じ構造で、中国語の値のみ日本語化（人名・作品名など固有名詞は原文の漢字のまま） |
| `instances_ja.jsonl` | 100インスタンスの対話文脈(STM)＋手がかりfacet題。`stm_zh`/`stm_ja`・`cued_title_zh`/`cued_title_ja` を併記 |
| `generations_ja.jsonl` | 決定版 N=100 実行の生成応答686件。`response_zh`/`response_ja` を併記（`_i` は元 `generations_clean.jsonl` の行番号） |

固有名詞は原文の漢字を保持し、台詞は各キャラの口調に合わせて訳しています。
構造（JSONのキー・配列長）は原文と一致するよう検証済みです。
