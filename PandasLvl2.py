# import pandas as pd
# import numpy as np
#
# data = {
#     'name': ['A','B','C','D', 'E'],
#     'age': [24, 30, np.nan, 40, 500],
#     'salary': [50000, np.nan, 60000, 90000, 100000],
#     'city': ['Delhi', 'Mumbai', 'Delhi', np.nan, 'Delhi']
# }
#
# df = pd.DataFrame(data)
#
# print("Original DF")
# print(df)
#
# print("\nINFO:")
# print(df.info())
#
# print("\nDESCRIBE:")
# print(df.describe())


#*******************Mini Demo of cleaning ***********************

import pandas as pd
import numpy as np

data = {
    'name': ['A','B','C','D', 'E'],
    'age': [24, 30, np.nan, 40, 500],
    'salary': [50000, np.nan, 60000, 90000, 100000],
    'city': ['Delhi', 'Mumbai', 'Delhi', np.nan, 'Delhi']
}

df = pd.DataFrame(data)

print("Original DF")
print(df)

# Missing values check
print("\nMissing values:")
print(df.isnull().sum())

# Fill missing numerical
df['age'] = df['age'].fillna(df['age'].median())
df['salary'] = df['salary'].fillna(df['salary'].mean())

# Fill missing categorical
df['city'] = df['city'].fillna(df['city'].mode()[0])

# Fix unrealistic age 500 -> remove
df = df[df['age'] < 100]

# Remove duplicates
df = df.drop_duplicates()

print("\nCleaned DF")
print(df)

print("\nINFO AFTER CLEANING:")
print(df.info())

print("\nDESCRIBE AFTER CLEANING:")
print(df.describe())
