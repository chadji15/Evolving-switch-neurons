import numpy as np
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt

with open(sys.argv[1], 'r') as fp:
    col_count = [ len(l.split()) for l in fp.readlines() ]

column_names = [i for i in range(0, max(col_count))]
df = pd.read_csv(sys.argv[1], header=None, delimiter=" ", names=column_names)
df = df.transpose()
df = df.fillna(method='ffill')
# df = df.transpose()

medians = df.median(axis=1)
lowq = df.quantile(q=0.25, axis=1)
hiq = df.quantile(q=0.75, axis=1)

plt.figure(figsize=(12,5))
plt.xlabel('Median, 1st and 3rd quantile of maximum fitness per generation')

ax1 = medians.plot(color='blue', grid=True)
ax2 = lowq.plot(color='red', grid=True, secondary_y=False)
ax3 = hiq.plot(color='red', grid=True, secondary_y=False)

plt.savefig(f"{sys.argv[1]}.png", bbox_inches='tight')

pass

