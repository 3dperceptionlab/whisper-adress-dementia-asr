import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def error_plot():
    df_control=pd.read_csv('resultados/transcriptionscc.csv')
    df_dementia=pd.read_csv('resultados/transcriptionscd.csv')
    error_cols=['repetitions', 'retracings', 'errors', 'unintelligibles', 'fragments']

    def normalize(df, group):
        for col in error_cols:
            df[col]=(df[col]/df['total words'])*100
            df['Group']=group
        return df

    df = pd.concat([normalize(df_control, 'Control'), normalize(df_dementia, 'Dementia')])

    df = df.melt(
        id_vars= ['ID', 'Group'],
        value_vars=error_cols,
        var_name='error_type',
        value_name='rate'
    )

    sns.set_context('talk', font_scale=1.2)
    sns.set_style('whitegrid')

    for error in error_cols:
        plt.figure(figsize=(6, 6))

        df_subset = df[df['error_type']==error]
        sns.violinplot(
            data = df_subset,
            x='Group', 
            y='rate',
            hue='Group',
            palette='Set2',
            inner=None,
            linewidth=0,
            alpha=0.5,
        )

        sns.stripplot(
            data=df_subset, 
            x='Group', 
            y='rate', 
            hue='Group', 
            palette='Set2', 
            jitter=True, 
            size=6, 
            legend=False   
        )

        colors = sns.color_palette('Dark2', n_colors=2)
        group_colors = {'Control': colors[0], 'Dementia': colors[1]}

        groups = ['Control', 'Dementia']
        for j, grp in enumerate(groups):
            grp_data = df_subset[df_subset['Group'] == grp]['rate']
            
            if len(grp_data) > 0:
                median = grp_data.median()
                q1 = grp_data.quantile(0.25)
                q3 = grp_data.quantile(0.75)
                
                plt.vlines(x=j, ymin=q1, ymax=q3, color=group_colors[grp], lw=4, zorder=10)
                plt.hlines(y=median, xmin=j-0.2, xmax=j+0.2, color=group_colors[grp], lw=4, zorder=10) 

        plt.title('')
        plt.xlabel('')
        plt.ylabel('')
        plt.xticks([])
        plt.tick_params(axis='y', labelsize=30)
        plt.ylim(bottom=-0.5, top=None)
        plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.1)

        plt.savefig('resultados/errores_'+error+'.pdf', format='pdf')
