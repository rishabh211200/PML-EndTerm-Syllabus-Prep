#We will use the last cleaned DataFrame,from last python file:

import pandas as pd
import numpy as np

data = {
    'name': ['A','B','C','D', 'E'],
    'age': [24, 30, np.nan, 40, 500],
    'salary': [50000, np.nan, 60000, 90000, 100000],
    'city': ['Delhi', 'Mumbai', 'Delhi', np.nan, 'Delhi']
}

df = pd.DataFrame(data)

# Fill missing numerical
df['age'] = df['age'].fillna(df['age'].median())
df['salary'] = df['salary'].fillna(df['salary'].mean())

# Fill missing categorical
df['city'] = df['city'].fillna(df['city'].mode()[0])

# Fix unrealistic age 500 -> remove
df = df[df['age'] < 100]

# Remove duplicates
df = df.drop_duplicates()

#Module 1

# print("Age > 30:")
# print(df[df['age'] > 30])
#
# print("\nAge >= 30 AND salary >= 60000:")
# print(df[(df['age'] >= 30) & (df['salary'] >= 60000)])
#
# print("\nCity = Delhi:")
# print(df[df['city'].isin(['Delhi'])])


#Module 2 - GroupBy

# print("Average salary per city:")
# print(df.groupby('city')['salary'].mean())
#
# print("\nCity-wise age median:")
# print(df.groupby('city')['age'].median())
#
# print("\nCity frequency:")
# print(df['city'].value_counts())


df['age_group'] = df['age'].apply(
    lambda x: 'young' if x < 30 else ('middle' if x <= 40 else 'senior')
)

df['is_high_salary'] = df['salary'].apply(lambda x: x > 70000)

#print(df)


# we created 2 Dataframes just for merging example:

df1 = pd.DataFrame({
    'id': [1,2,3],
    'name': ['A','B','C']
})

df2 = pd.DataFrame({
    'id': [1,2,4],
    'salary': [50000, 75000, 90000]
})

merged = pd.merge(df1, df2, on='id', how='inner')
print("Inner join:")
print(merged)

merged_left = pd.merge(df1, df2, on='id', how='left')
print("\nLeft join:")
print(merged_left)

merged_right = pd.merge(df1, df2, on='id', how='right')
print("\nRight join:")
print(merged_right)

merged_outer = pd.merge(df1, df2, on='id', how='outer')
print("Outer join:")
print(merged_outer)
