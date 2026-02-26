import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def pauses_plot():
    df_control = pd.read_csv('resultados/cc.csv')
    df_dementia = pd.read_csv('resultados/cd.csv')

    df_control['Group'] = 'Control'
    df_dementia['Group'] = 'Dementia'

    df_final = pd.concat([df_control, df_dementia])

    df = df_final[df_final['Pause duration']>0.1].copy()

    limits = [0.1, 0.5, 1.0, 1.5, np.inf]
    labels = ['0.1 - 0.5s', '0.5 - 1.0s', '1.0 - 1.5s', '> 1.5s']

    df['Range'] = pd.cut(df['Pause duration'], bins=limits, labels=labels, right=False)
    df_sum = df.groupby(['Group', 'Range'], observed=False)['Pause duration'].sum().reset_index()

    sns.set_context('talk', font_scale=1.2)
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 6))

    sns.countplot(
        data=df,
        x='Range',
        hue='Group',
        palette='Set2',
        edgecolor='black', 
        linewidth=1.5,
        width=0.6,
        legend=False
    )

    plt.title('')
    plt.xlabel('')
    plt.ylabel('')

    plt.savefig('resultados/pausas.pdf', format='pdf', bbox_inches='tight')