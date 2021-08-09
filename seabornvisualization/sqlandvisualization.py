# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 15:25:42 2021

@author: asrinutku
"""
import pandas as pd
import pyodbc 
import numpy as np
import seaborn as sns

conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=DESKTOP-M006Q3T;'
                      'Database=AtıklarDb;'
                      'Trusted_Connection=yes;')

cursor = conn.cursor()

df_Waste = pd.read_sql_query('SELECT * FROM AtıklarDb.dbo.Waste',conn)
df_DayType = pd.read_sql_query('SELECT * FROM AtıklarDb.dbo.DayType',conn)
df_DailyWaste = pd.read_sql_query('SELECT * FROM AtıklarDb.dbo.DailyWaste',conn)

formatted_Sds = df_DailyWaste["Sds"].dt.strftime("%m/%d/%y")
formatted_Sde = df_DailyWaste["Sde"].dt.strftime("%m/%d/%y")

df_DailyWaste.drop(['Sds', 'Sde'], axis = 1,inplace = True)

ext_col = formatted_Sds
ext_col1 = formatted_Sde

df_DailyWaste.insert(1,"Sds",ext_col)
df_DailyWaste.insert(2,"Sde",ext_col1)



        
#%%
DaysDct = df_DayType.to_dict('split')


for i in df_DailyWaste['DayType']:
    for j in DaysDct["data"]:
        if(i in j):
            df_DailyWaste['DayType'] = df_DailyWaste['DayType'].replace([i],j[1])
            break
        
#%%

w =[]
#for i in df_DailyWaste['Wasteid']:
for i in df_DailyWaste['Wasteid']:
    w.append(df_Waste.loc[df_Waste['id'].isin([i])]["AmountofWaste"].values)
    
for i in range(len(w)):

    w[i]= (int((str(w[i]).lstrip('[').rstrip(']'))))

df_DailyWaste.drop("Wasteid", axis = 1, inplace = True)
df_DailyWaste["Waste"] = w



#%%
sns.set_style("darkgrid")
sns.set_context("poster", font_scale = .4, rc={"grid.linewidth": 0.5})
sns.set(rc = {'figure.figsize':(15,8)})
sns.distplot(df_Waste.AmountofWaste,
             bins ='auto',
             rug= False,
             color ="r",
             kde=0,
             ).set(title='Atık Miktarı')


#%%
sns.kdeplot(df_Waste.AmountofWaste,
            shade = True,
            color ="r",
            ).set(title='Atık Miktarı Yoğunluğu')

#%%

sns.lineplot(data=df_DailyWaste,
             x="Sds",
             y="Waste",
             color="r"
             ).set(title='Özel Günler Atık Miktarı Karşılaştırması')


#%%

import matplotlib.pyplot as plt

myexp = [0,0,0,0,0,0,0,0,0]
lb =df_DailyWaste.DayType

plt.pie(w,
        labels=lb,
        explode = myexp,
        
autopct='%1.1f%%', shadow=True, startangle=140)


