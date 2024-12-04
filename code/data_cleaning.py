import pandas as pd

df_names_org = pd.read_csv('../data/forenames_original.csv', low_memory=False)

num_forenames = len(df_names_org)

df_males = df_names_org[df_names_org['gender'] == 'M'].sample(n=50000, random_state=42, replace=False)
df_females = df_names_org[df_names_org['gender'] == 'F'].sample(n=50000, random_state=42, replace=False)
df_names = pd.concat([df_males, df_females]).reset_index(drop=True)

df_names = df_names.drop(columns=['country', 'count'])

df_names = df_names[df_names['gender'].isin(['F', 'M'])]

df_names.dropna(subset=['forename'], inplace=True)

df_names.reset_index(drop=True, inplace=True)

df_names['forename'] = df_names['forename'].fillna('')

df_names['gender'] = df_names['gender'].map({'M': 0, 'F': 1})

df_names.to_csv('../data/names.csv', index = False)