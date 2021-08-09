# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 09:48:51 2021

@author: asrinutku
"""
import pandas as pd
import seaborn as sns

xls = pd.ExcelFile('2021_bcekmece_tekstil.xlsx')
months = ["OCAK","ŞUBAT","MART","NİSAN","MAYIS","HAZİRAN"]
waste =[]
x=1

d = {}
for x in range(1, 7):
    d["df_ay{0}".format(x)] = pd.read_excel(xls , months[x-1])
    
list_of_dict_values = list(d.values())
d.clear()

for df in list_of_dict_values:
    df = df.rename(columns={
                            'Unnamed: 1': 'TARİH',
                            'Unnamed: 2': 'ATIK',
                            'Unnamed: 3': 'KONTEYNER',
                            })
    
    df = df.drop(columns=['KONTEYNER','BÜYÜKÇEKMECE BELEDİYESİ TEKSTİL ATIĞI-EKİPMAN LİSTESİ'])
    df = df.iloc[3:]
    df = df.dropna()
    
    d["df_ay{0}".format(x)] = df
    x +=1

list_of_dict_values = list(d.values())
d.clear()
    
for i in list_of_dict_values:
    waste.append(i["ATIK"].sum())

#%%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams['figure.figsize'] = (20, 10)
y_pos = np.arange(len(months))



plt.bar(y_pos, 
        waste,
        color=["r"],
        edgecolor='black',
        )

plt.grid(color = 'black', linestyle = '--', linewidth = 0.5 , axis='y')


plt.xticks(y_pos, months)
plt.xlabel("AYLAR")
plt.ylabel("TEKSTİL ATIK MİKTARI")

plt.show()







    