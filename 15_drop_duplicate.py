# -*- coding: utf-8 -*-
# Auto-generated from 15_drop_duplicate.ipynb
# Cells are delimited with '# %%' markers.

# %% [markdown]
## 複数フェーズで同じ事業評価書になっているレコードを重複排除する

# %%
import pandas as pd
df_ori = pd.read_csv('df_check_pro.csv')

count = len(df_ori)
print(f"レコード数: {count}")

# %%
def add_sequential_number_groupby(df, key_column, new_column_name='sequential_id'):
  """
  特定のカラムをキーに連番を付与する

  Args:
    df (pd.DataFrame): 対象のDataFrame
    key_column (str): キーとなるカラム名
    new_column_name (str): 新しい連番カラム名

  Returns:
    pd.DataFrame: 連番カラムが追加されたDataFrame
  """
  df[new_column_name] = df.groupby(key_column, dropna=False).cumcount(ascending=False) + 1
  return df

# keyが同じものを重複プロジェクトとして削除する
# webサイトのつくり上、フェーズ１と２が分かれて掲載されているが、どちらも同じPDFファイルの場合重複を削除する
key = ['国名','プロジェクト期','プロジェクトコスト_計画時_int', 'プロジェクトコスト_実績_int','プロジェクト期間開始_計画時','プロジェクト期間終了_計画時', 'file']
df = df_ori.sort_values(key)

key.remove('file')
df = add_sequential_number_groupby(df, key, new_column_name='連番')
df = df[df['連番']==1]
count = len(df)
print(f"レコード数: {count}")

# %%
# 除外されたものを確認
outer_join = df_ori.merge(df, how = 'outer', indicator = True)
anti_join = outer_join[~(outer_join._merge == 'both')].drop('_merge', axis = 1)
# 削除件数一致の確認
assert len(anti_join) == len(df_ori) - len(df)
anti_join.head(5)

# %%
# 重複レコードを結合して除外
df_check = df_ori.merge(anti_join, on=key, how = 'inner')
df_check.to_csv("duplicate_check_pro.csv")

# %%
df.to_csv("df_check_3.csv")

