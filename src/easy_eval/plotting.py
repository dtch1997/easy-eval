import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")

def models_plot(df: pd.DataFrame, column: str):
    plt.figure(figsize=(10, 6))
    sns.barplot(x="model", y=column, data=df)
    plt.show()