# -*- coding: utf-8 -*-
# Auto-generated from 99_rename_df.ipynb
# Cells are delimited with '# %%' markers.

# %% [markdown]
## モデル作成用に前処理をする

# %%
import pandas as pd
df = pd.read_csv("df_check_12.csv", index_col=0)

# %%
for c in df.columns:
    print(c)

# %%
rename = {
    '評価年度':'eval_year',
    '評価会社':'evaluator_group',
    '評価者':'evaluator',
    'プロジェクトコスト_計画時_int':'project_cost_plan',
    'プロジェクトコスト_実績_int':'project_cost_act',
    '事前評価時_プロジェクト期間（月）':'project_duration_plan', 
    '実績_プロジェクト期間（月）':'project_duration_act', 
    '地域詳細':'region_detail',
    '地域':'region',
    '事業形態':'type',
}
df=df.rename(columns=rename)
df

# %%
df['total_eval'].unique()

# %%
df.to_csv('df_check_99.csv')

# %%
for c in df.columns.tolist():
    print(c)

# %%

