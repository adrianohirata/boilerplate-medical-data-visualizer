import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv("medical_examination.csv", index_col="id")

#2
df['BMI'] = df["weight"] / ((df["height"]/100) ** 2)

#df['overweight'] = np.where(df['BMI'] > 25, 1, 0)
df['overweight'] = df['BMI'].apply(lambda x: 1 if x > 25 else 0)

# drop aux column BMI
df.drop('BMI', axis=1, inplace=True)

# 3
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)

df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4
def draw_cat_plot():
    # 5
    df_cat = df.melt(id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6
    df_cat = df_cat.sort_values(by='variable')

    df_cat = df_cat.groupby(['cardio', 'variable', 'value'])['value'].size().reset_index(name='total')
  
    # 7
    bar_plot = sns.catplot(data=df_cat, col='cardio', x='variable', y='total', hue='value', kind='bar')

    # 8
    fig = bar_plot.fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
        ]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(14, 12))

    # 15
    sns.color_palette("icefire", as_cmap=True)
    sns.heatmap(corr, mask=mask, ax=ax, annot=True, fmt='.1f', center=0)

    # 16
    fig.savefig('heatmap.png')
    return fig
