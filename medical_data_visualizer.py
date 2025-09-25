import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import the data
df = pd.read_csv("medical_examination.csv")

# 2. Add overweight column
df['BMI'] = df['weight'] / ((df['height']/100) ** 2)
df['overweight'] = (df['BMI'] > 25).astype(int)

# 3. Normalize data
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4. Draw Categorical Plot
def draw_cat_plot():
    df_cat = pd.melt(df, id_vars=['cardio'], 
                     value_vars=['cholesterol','gluc','smoke','alco','active','overweight'])
    df_cat = df_cat.groupby(['cardio','variable','value']).size().reset_index(name='total')
    
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio',
                      data=df_cat, kind='bar').fig
    return fig

# 5. Draw Heat Map
def draw_heat_map():
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

    corr = df_heat.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', cmap='coolwarm', square=True, ax=ax)
    return fig

# Example usage
if __name__ == "__main__":
    cat_fig = draw_cat_plot()
    cat_fig.savefig("catplot.png")

    heat_fig = draw_heat_map()
    heat_fig.savefig("heatmap.png")
    print("Plots saved as catplot.png and heatmap.png")
