# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 17:03:15 2021

@author: asrinutku
"""
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('bcekmece_2donem_yag.csv')
df.drop([0,1,72], inplace = True)


df = df.rename(columns={'Unnamed: 0': 'id',
                            'Unnamed: 1': 'mekan',
                            'Unnamed: 2': 'del',
                            'Unnamed: 3': 'nisan',
                            'Unnamed: 4': 'mayis',
                            'Unnamed: 5': 'haziran'})

df.drop(['del'], axis = 1,inplace = True)

t = df.iloc[-1,2:5]

df_v = pd.DataFrame({'amount':t.values,'months':t.index})
 
df_v["amount"] = pd.to_numeric(df_v["amount"])

#%%
sns.lineplot(data=df_v,
             x="months",
             y="amount",
             color="r"
             ).set(title='3 Aylık Atık Miktarı Karşılaştırması')

#%%
sns.barplot(
    x="months",
    y="amount",
    data=df_v,
    color="r"
    ).set(title='3 Aylık Atık Miktarı Karşılaştırması')