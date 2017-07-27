import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./data/loss', header=None)
df[0].plot(figsize=(8, 6))
plt.show()
