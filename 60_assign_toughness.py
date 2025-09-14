# -*- coding: utf-8 -*-
# Auto-generated from 60_assign_toughness.ipynb
# Cells are delimited with '# %%' markers.

# %% [markdown]
## 評価者の総合評価平均値を算出する

# %%
import pandas as pd
df = pd.read_csv("df_check_11.csv", index_col=0)
df

# %%
li = df['評価会社'].unique().tolist()
li

# %%
def replace_value(df: pd.DataFrame, colname: str, li: list, repword) -> pd.DataFrame:
    """
    dfのcolnameのカラムについて、liで設定されたリストの値をrepwordの値に置換する
    """
    out = df.copy()
    out[colname] = out[colname].replace(li, repword)
    return out

# %%
li=[
 '(株)グローバル・グループ 21 ジャパン',
 '（株）グローバル・グループ 21 ジャパン',
 '(株)グローバル・グループ21 ジャパン',
 '(株)グローバル・グループ21ジャパン',
 '株式会社グローバル・グループ 21 ジャパン',
 '株式会社グローバル・グループ21 ジャパン',
 '株式会社グローバル・グループ21ジャパン',
 '株式会社グローバルグループ21ジャパン',
]
df = replace_value(df=df, colname="評価会社", li=li, repword="株式会社グローバル・グループ21ジャパン")

# %%
li=[
 '(株) タックインターナショナル',
 '(株)タックインターナショナル',
 'タック・インターナショナル',
 '株式会社タックインターナショナル',
]
df = replace_value(df=df, colname="評価会社", li=li, repword="株式会社タック・インターナショナル")

# %%
li=[
 'EY 新日本サステナビリティ(株)',
 'EY 新日本有限責任監査法人',
 'EYアドバイザリー・アンド・コンサルティング株式会社',
 'EY新日本有限責任監査法人',
 'アーンスト・アンド・ヤング・アドバイザリー(株)',
 'アーンスト・アンド・ヤング・アドバイザリー株式会社',
 '新日本サステナビリティ(株)',
 '新日本サステナビリティ株式会社',
 '新日本有限責任監査法人'
]
df = replace_value(df=df, colname="評価会社", li=li, repword="EY 新日本サステナビリティ株式会社")

# %%
li=[
 '(一般財団法人)国際開発機構',
 '一般財団法人 国際開発機構',
 '一般財団法人国際開発機構',
]
df = replace_value(df=df, colname="評価会社", li=li, repword="一般財団法人国際開発機構")

# %%
li=[
 '(株)国際開発センター',
 '株式会社 国際開発センター',
 '国際開発センター',
]
df = replace_value(df=df, colname="評価会社", li=li, repword="株式会社国際開発センター")

# %%
li=[
 'アイ・シー・ネット(株)',
 'アイシーネット株式会社',
]
df = replace_value(df=df, colname="評価会社", li=li, repword="アイ・シー・ネット株式会社")

# %%
li=[
 '三菱 UFJ リサーチ&コンサルティング株式会社',
 '三菱 UFJリサーチ&コンサルティング株式会社',
 '三菱UFJリサーチ&コンサルティング',
]
df = replace_value(df=df, colname="評価会社", li=li, repword="三菱UFJリサーチ&コンサルティング株式会社")

# %%
li=[
 '三菱 UFJ リサーチ&コンサルティング株式会社, PB ジャパン株式会社',
 '三菱UFJリサーチ&コンサルティング株式会社, PB ジャパン株式会社',
 '三菱UFJリサーチ&コンサルティング株式会社',
]
df = replace_value(df=df, colname="評価会社", li=li, repword="三菱UFJリサーチ&コンサルティング株式会社, PBジャパン株式会社")

# %%
li=[
 '三菱 UFJ リサーチ&コンサルティング株式会社, PB ジャパン株式会社',
 '三菱UFJリサーチ&コンサルティング株式会社, PB ジャパン株式会社',
 '三菱UFJリサーチ&コンサルティング株式会社',
]
df = replace_value(df=df, colname="評価会社", li=li, repword="三菱UFJリサーチ&コンサルティング株式会社, PBジャパン株式会社")

# %%
li=[
 'アルメック VPI',
 '株式会社アルメック VPI',
 '株式会社アルメックVPI',
]
df = replace_value(df=df, colname="評価会社", li=li, repword="株式会社アルメック")

# %%
li=[
 'Value Frontier 株式会社',
 'Value Frontier(株)',
]
df = replace_value(df=df, colname="評価会社", li=li, repword="Value Frontier 株式会社")

# %%
li=[
 '（財）国際開発高等教育機構',
]
df = replace_value(df=df, colname="評価会社", li=li, repword="財団法人国際開発高等教育機構")

# %%
li=[
 'インテムコンサルティング',
 'インテムコンサルティング株式会社',
]
df = replace_value(df=df, colname="評価会社", li=li, repword="インテムコンサルティング株式会社")

# %%
li=[
 '株式会社アースアンドコーポレーション',
]
df = replace_value(df=df, colname="評価会社", li=li, repword="株式会社アースアンドヒューマンコーポレーション")

# %%
li=[
 'コーエイ総合研究所',
]
df = replace_value(df=df, colname="評価会社", li=li, repword="株式会社コーエイ総合研究所")

# %%
li=[
 '株式会社日本経済研究所',
 '日本経済研究所',
]
df = replace_value(df=df, colname="評価会社", li=li, repword="株式会社日本経済研究所")

# %%
# jica事務所の表記揺れ修正
df['評価会社'] = df['評価会社'].astype(str).str.replace('東チモール事務所', '東ティモール事務所', regex=False)
df['評価会社'] = df['評価会社'].astype(str).str.replace('中華人民共和国事務所', '中国事務所', regex=False)
df['評価会社'] = df['評価会社'].astype(str).str.replace('JICA', '', regex=False)
df['評価会社'] = df['評価会社'].astype(str).str.replace(' ', '', regex=False)
df['評価会社'] = df['評価会社'].astype(str).str.replace('　', '', regex=False)
df['評価会社'] = df['評価会社'].astype(str).str.replace('民主共和国', '', regex=False)
df['評価会社'] = df['評価会社'].astype(str).str.replace('共和国', '', regex=False)
df['評価会社'] = df['評価会社'].astype(str).str.replace('支所', '事務所', regex=False)
df['評価会社'] = df['評価会社'].astype(str).str.replace('駐在員', '', regex=False)

# %%
no_replace = ['中国事務所', '中華人民共和国事務所']

df['評価会社'] = df['評価会社'].apply(
    lambda s: (
        s
        if isinstance(s, str) and s in no_replace
        else (s.replace('国事務所', '事務所') if isinstance(s, str) else s)
    )
)

# %%
col = '評価会社' 
unique_sorted = sorted(df[col].dropna().unique())
unique_sorted

# %%
import matplotlib.pyplot as plt
import japanize_matplotlib

# 総合評価を数値に変換
df['total_eval'] = df['総合評価'].map({'非常に高い': 4, '高い': 3, '中程度': 2,'一部課題がある': 2, '一部に課題があると判断される': 2, '低い': 1,'非常に低い': 1})


df_agg = df.groupby('評価会社')['total_eval'].agg(['mean', 'count']).reset_index()

# Rename columns for clarity
df_agg.columns = ['Evaluator Group', 'Average Total Eval', 'Count']

# Sort for better visualization
df_agg = df_agg.sort_values('Average Total Eval')

# Create a bar chart of the average total eval
plt.figure(figsize=(50, 30))
plt.bar(df_agg['Evaluator Group'], df_agg['Average Total Eval'])
plt.xlabel('評価者グループ')
plt.ylabel('合計評価の平均')
plt.title('評価者グループごとの合計評価平均')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Print the aggregated dataframe
print("Evaluator Groupごとの合計評価の平均と集約数:")
df_agg

# %%
print(df_agg.columns.tolist())

# %%

def calculate_group_mean_excluding_self(row, df, group_col, value_col):
  """
  Calculates the mean of 'value_col' for the group in 'group_col' that the row belongs to,
  excluding the value from the current row.
  """
  group_data = df[df[group_col] == row[group_col]]
  sum_excluding_self = group_data[value_col].sum() - row[value_col]
  count_excluding_self = group_data[value_col].count() - 1

  if count_excluding_self > 0:
    return sum_excluding_self / count_excluding_self
  else:
    # Handle the case where the group only has one member (the current row)
    return None

df['avg_outcome_evalator'] = df.apply(
    lambda row: calculate_group_mean_excluding_self(row, df, '評価会社', 'total_eval'),
    axis=1
)

df


# %%
# 検算
col=['評価会社','案件名','total_eval', 'avg_outcome_evalator']
df[df['評価会社']=='ガボン事務所'][col]

# %%
df.to_csv("df_check_12.csv")

# %%

# %%

