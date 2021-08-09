# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 10:51:16 2021

@author: asrinutku
"""
import pandas as pd
import seaborn as sns

xls = pd.ExcelFile('2021_bcekmece_cam_2.xlsx')
a=0 #loop 
deleteempty = []
wasteamounts = []
dates = []


d = {}
for x in range(1, 13):
    d["df_ay{0}".format(x)] = pd.read_excel(xls , '{0}.AY'.format(x))
    
list_of_dict_values = list(d.values())


for i in list_of_dict_values:
    if i.empty:
        deleteempty.append(a)
    a += 1
del list_of_dict_values[deleteempty[0]:deleteempty[-1]+1]


for i in list_of_dict_values:
    i["TARİH"] = i["TARİH"].dt.strftime("%m/%d/%y")

for i in list_of_dict_values:
    wasteamounts.append(i["MİKTAR"].iloc[-1])
    dates.append(i["TARİH"].iloc[1])
    
#%%
import matplotlib.pyplot as plt

sns.set_style("darkgrid")
# sns.set_context("poster", font_scale = .4, rc={"grid.linewidth": 3.5})
sns.set(rc = {'figure.figsize':(15,8)})

sns.lineplot(
             x=dates,
             y=wasteamounts,
             color="r",
             linewidth="3.5",
             ).set(title='Aylık Cam Atığı')

plt.xlabel("AYLAR")
plt.ylabel("ATIK MİKTARI")

#%%
import numpy as np
import matplotlib

matplotlib.rcParams['figure.figsize'] = (20, 10)
y_pos = np.arange(len(dates))

plt.bar(y_pos, 
        wasteamounts,
        color=["r"],
        edgecolor='black',
        )

plt.xticks(y_pos, dates)
plt.xlabel("AYLAR")
plt.ylabel("ATIK MİKTARI")

plt.show()

# %%
vehicles = []
for i in list_of_dict_values:
    vehicles.append(i.iloc[:,[True, False, False,True, True]])
    

vehicles_df = pd.concat(vehicles)

vehicles_df = vehicles_df.dropna()

vehicles_df.drop(vehicles_df.loc[vehicles_df['PLAKA']=='TOPLAM'].index, inplace=True)

for i in vehicles_df.iloc[:,1]:
    
    new_i = i.replace(" ", "")
    vehicles_df['PLAKA'] = vehicles_df['PLAKA'].replace([i],new_i)


#%%
import matplotlib.pyplot as plt

# print(vehicles_df.groupby('PLAKA').sum()[['MİKTAR']])
sns.set_style("darkgrid")
sns.set_context("poster", font_scale = .5, rc={"grid.linewidth": 0.5})
sns.set(rc = {'figure.figsize':(15,8)})


sns.barplot(x="PLAKA",
            y="MİKTAR", 
            data=vehicles_df,
            palette="Blues_d"
            ).set(title='Araçların Topladıkları Atık Miktarı')

plt.xlabel("ARAÇLAR")
plt.ylabel("ATIK MİKTARI")

plt.show()

#%%

sns.kdeplot(vehicles_df.MİKTAR,
            shade = True,
            color="r",
            ).set(title='ATIK MİKTARI YOĞUNLUĞU')

#%%

(sns
 .FacetGrid(vehicles_df,
              hue = "PLAKA",
              height = 7,
              xlim = (0, 10000))
 .map(sns.kdeplot, "MİKTAR", shade= True)
 .add_legend()
);

#%%

sns.set_style("darkgrid")
sns.set_context("poster", font_scale = .5, rc={"grid.linewidth": 0.5})
sns.set(rc = {'figure.figsize':(30,8)})


sns.scatterplot(x = "TARİH",
            y = "MİKTAR",
            s=1000,
            hue = "PLAKA",
            data = vehicles_df,

            );

plt.xlabel("ARAÇLAR")
plt.ylabel("ATIK MİKTARI")

plt.show()













