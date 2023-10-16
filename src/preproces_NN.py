from src.utils import *

import pandas as pd

df = pd.read_parquet('data/raw_data/df.parquet.gzip')
df = df.dropna()

new_df = pd.DataFrame(columns=['code', 'language'])
languages = df.language.unique()
for l in tqdm(range(len(languages))):
    x = df.loc[df['language'] == languages[l]]["code"]
    for row in x:
        y = RemoveComments(row, languages[l])
        new_row = {'code': y, 'language': languages[l]}
        new_df = new_df._append(new_row, ignore_index=True)

new_df.to_parquet('data/preprocessed_data/dataWithoutComments.parquet')
