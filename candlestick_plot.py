import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("./crypto_data/BCH-USD-resample-1D.csv")
df_candle = df[["date", "open", "high", "low", "close"]]

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.vlines(df.date, df.open, df.close)
ax.xaxis.set_major_locator(plt.MaxNLocator(8))
ax.set_xticklabels(df.date, rotation=60)

plt.show()
