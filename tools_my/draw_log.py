import pandas as pd
import matplotlib.pyplot as plt


csv_path = r'G:\program\CD\ddmp-cd-my\out_dir\levir_basic_unet\progress.csv'
df = pd.read_csv(csv_path)
loss = df['loss'].values
step = df['step'].values
start = 0
plt.plot(step[start:], loss[start:])
plt.show()
