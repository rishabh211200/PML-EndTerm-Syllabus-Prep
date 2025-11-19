import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a = np.array([1,2,3])
print(a)

df = pd.DataFrame({'x':[1,2,3],'y':[10,20,30]})
print(df)

plt.plot(df['x'], df['y'])
plt.show()