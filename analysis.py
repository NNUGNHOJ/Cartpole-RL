import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#'3', '6', '12', '24', '48', '96', '192'

n_binss = (24, 4)
print(n_binss.head())


df = pd.read_pickle('tabular_bins_experiment_2')

index = []
for i in range(0, 1000):
    index.append(i)

df['index'] = pd.Series(index)

df.plot(x='index', y=["12"], kind="line", lw=0.4)
plt.show()
