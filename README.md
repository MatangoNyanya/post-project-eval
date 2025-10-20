# Post Project Evaluation

このディレクトリは、事業評価データの収集・整形・特徴量付与を段階的に行う プログラム群です。mainを実行すると、原データの取得から学習用のデータセット整備までを再現できます。
中間ファイルも含めて公開していますが、最終出力だけ参照したい場合は、以下を参照してください。
- df_check_pro.csv: Geminiでの抽出結果を保存したデータセット
- df_check_99.csv: 各種整形及び世銀など外部サービスから取得した特徴量を追加したデータセット
- df_check_ml.csv: 成功・失敗を予測する機械学習用のデータセット

また、論文中でサンプリングチェックを実施した際のチェック対象は以下のデータセットです。
- df_sample.csv

## ノートブックの流れ
- `00_main.py`: 実行順に沿ってプログラムを全実行する機能。ただし、API呼び出し箇所は長時間実行となるため、別実行が推奨。
- `05_get_data_from_api.py`: データ取得（API/スクレイピング等）。必要に応じてAPIキーは環境変数から読み込んでください。
- `15_drop_duplicate.py`: 複数フェーズで同じ事業評価書を指定している場合に作成される重複レコードの除去。
- `20_calc_cost_duration.py`: コスト・期間の算出と補完。
- `25_assign_region.py`: 国に紐づく地域情報の付与。
- `30_assign_cat_dummy.py`: 支援分野などカテゴリのダミー化。
- `35_assign_wgi.py`: 世界銀行が公開する指標（WGI）の付与。
- `40_assign_freedomrate.py`: Freedom House 指標の付与。
- `45_assign_gdp.py`: GDP/Growth の付与。
- `50_assign_population.py`: 人口指標の付与。
- `55_assign_ex_eval_flg.py`: 外部評価フラグの付与。
- `60_assign_toughness.py`: 評価者の厳しさに関する特徴量の付与。
- `99_rename_df.py`: 列名整理など最終整形。

## 必要環境
- Python 3.9 以上（推奨: 3.10）
- 主要パッケージ（例）
  - pandas, numpy, matplotlib
  - requests, beautifulsoup4
  - その他、各ノートの冒頭セルに記載のパッケージ

## 既知の注意点
- APIキーやGCPの環境については、各自作成したものを設定してください。

## ディレクトリ構成（抜粋）
```
post-project-eval/
  ├── 05_get_data_from_api.py
  ├── 15_drop_duplicate.py
  ├── 20_calc_cost_duration.py
  ├── 25_assign_region.py
  ├── 30_assign_cat_dummy.py
  ├── 35_assign_wgi.py
  ├── 40_assign_freedomrate.py
  ├── 45_assign_gdp.py
  ├── 50_assign_population.py
  ├── 55_assign_ex_eval_flg.py
  ├── 60_assign_toughness.py
  └── 99_rename_df.py
```

## 依存関係 / Requirements

プロジェクトの Python 依存パッケージは `requirements.txt` に記載されています。基本的な実行に必要なパッケージの例は以下です:

- nbformat
- nbclient
- pandas

インストール手順（macOS / bash）:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```
