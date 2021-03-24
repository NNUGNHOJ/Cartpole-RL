import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#'3', '6', '12', '24', '48', '96', '192'


df = pd.read_pickle('tabular_bins_experiment')

index = []
for i in range(0, 1000):
    index.append(i)

df['index'] = pd.Series(index)

df.plot(x='index', y=["3"], kind="line", lw=0.4)
plt.show()
