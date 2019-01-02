import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

df = pd.read_csv('/home/sarvesh/ML_Github/hbwid/train.csv')
print(len(df['Id'].value_counts()))
#print(df['Id'].value_counts())
le = LabelEncoder()
df['Id_numeric'] = le.fit_transform(df['Id'])
df_sorted = df.sort_values('Image')
df_sorted.drop(['Image', 'Id'], axis = 1, inplace = True)
print(df_sorted.head())