# Post Project Evaluation Notebooks

このディレクトリは、事業評価データの収集・整形・特徴量付与を段階的に行う Jupyter Notebook 群です。ノートブックを上から順に実行することで、原データの取得から学習用のデータセット整備までを再現できます。

## ノートブックの流れ
- `05_get_data_from_api.ipynb`: データ取得（API/スクレイピング等）。必要に応じてAPIキーは環境変数から読み込んでください。
- `15_drop_duplicate.ipynb`: 複数フェーズで同じ事業評価書を指定している場合に作成される重複レコードの除去。
- `20_calc_cost_duration.ipynb`: コスト・期間の算出と補完。
- `25_assign_region.ipynb`: 国に紐づく地域情報の付与。
- `30_assign_cat_dummy.ipynb`: 支援分野などカテゴリのダミー化。
- `35_assign_wgi.ipynb`: 世界銀行が公開する指標（WGI）の付与。
- `40_assign_freedomrate.ipynb`: Freedom House 指標の付与。
- `45_assign_gdp.ipynb`: GDP/Growth の付与。
- `50_assign_population.ipynb`: 人口指標の付与。
- `55_assign_ex_eval_flg.ipynb`: 外部評価フラグの付与。
- `60_assign_toughness.ipynb`: 評価者の厳しさに関する特徴量の付与。
- `99_rename_df.ipynb`: 列名整理など最終整形。

## 必要環境
- Python 3.9 以上（推奨: 3.10）
- Jupyter Lab/Notebook
- 主要パッケージ（例）
  - pandas, numpy, matplotlib
  - requests, beautifulsoup4（データ取得が必要な場合）
  - その他、各ノートの冒頭セルに記載のパッケージ

## 既知の注意点
- ノート出力（セルの outputs）は公開用にクリア済みです。初回実行時はセルを順に実行し、必要パッケージは都度インストールしてください。
- APIキーやGCPの環境については、各自作成したものを設定してください。

## ディレクトリ構成（抜粋）
```
post-project-eval/
  ├── 05_get_data_from_api.ipynb
  ├── 15_drop_duplicate.ipynb
  ├── 20_calc_cost_duration.ipynb
  ├── 25_assign_region.ipynb
  ├── 30_assign_cat_dummy.ipynb
  ├── 35_assign_wgi.ipynb
  ├── 40_assign_freedomrate.ipynb
  ├── 45_assign_gdp.ipynb
  ├── 50_assign_population.ipynb
  ├── 55_assign_ex_eval_flg.ipynb
  ├── 60_assign_toughness.ipynb
  └── 99_rename_df.ipynb
```