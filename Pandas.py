# 🔶 MODULE 1: PANDAS (Basics to Exam-Level)
# ⭐ Step 1 — What is Pandas?

# Pandas = Python ka Excel + SQL + NumPy ka combination
# It gives you:

# Series → ek column

# DataFrame → pura table (rows × columns)

# Example:
# Name | Age | Salary
# A      24     50000
# B      30     80000
# Ye sab Pandas DataFrame me aata hai.

#import pandas:
import pandas as pd

#
# # ⭐ Step 3 — Common Exam Functionality
#
# # Below is EXACTLY what you NEED for exam:
#
# # ✔ Loading data
# df = pd.read_csv("data.csv")
#
# # ✔ Inspecting data
# df.head()        # first 5 rows
# df.tail()        # last 5 rows
# df.info()        # data types
# df.describe()    # statistical summary
#
# # ✔ Selecting columns
# df['age']              # single column
# df[['age','salary']]   # multiple columns
#
# # ✔ Selecting rows (filtering)
# df[df['age'] > 25]
# df[(df['age'] > 25) & (df['salary'] < 50000)]
#
# # ✔ Adding new column
# df['bonus'] = df['salary'] * 0.10
#
# # ✔ Dropping columns
# df = df.drop('bonus', axis=1)
#
# # ✔ Handling missing values
# df.dropna()               # drop rows with NA
# df.fillna(0)              # replace NA with 0
# df['age'].fillna(df['age'].mean())
#
# # ✔ Sorting
# df.sort_values('salary', ascending=False)
#
# # ✔ GroupBy (super important)
# df.groupby('department')['salary'].mean()

import pandas as pd

data = {
    'name': ['A','B','C','D'],
    'age': [24, 30, 29, 40],
    'salary': [50000, 80000, 60000, 90000]
}

df = pd.DataFrame(data)

print(df)
print(df.head())
print(df.info())
print(df.describe())
print(df[['name','salary']])
print(df[df['age'] > 28])



#**********************Output***********************

#   name  age  salary
# 0    A   24   50000
# 1    B   30   80000
# 2    C   29   60000
# 3    D   40   90000
#   name  age  salary
# 0    A   24   50000
# 1    B   30   80000
# 2    C   29   60000
# 3    D   40   90000
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 4 entries, 0 to 3
# Data columns (total 3 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   name    4 non-null      object
#  1   age     4 non-null      int64
#  2   salary  4 non-null      int64
# dtypes: int64(2), object(1)
# memory usage: 228.0+ bytes
# None
#             age        salary
# count   4.00000      4.000000
# mean   30.75000  70000.000000
# std     6.70199  18257.418584
# min    24.00000  50000.000000
# 25%    27.75000  57500.000000
# 50%    29.50000  70000.000000
# 75%    32.50000  82500.000000
# max    40.00000  90000.000000
#   name  salary
# 0    A   50000
# 1    B   80000
# 2    C   60000
# 3    D   90000
#   name  age  salary
# 1    B   30   80000
# 2    C   29   60000
# 3    D   40   90000

# 🔥 Part 1 — Tumne jo 4 outputs diye hain, unka meaning / interpretation
# 1st Output:
# name  age  salary
# A     24   50000
# B     30   80000
# C     29   60000
# D     40   90000
#
#
# 👉 This is your DataFrame — like an Excel table.
# Each column = Pandas Series.
# Each row = record/observation.
#
# name is categorical
#
# age is numerical (int)
#
# salary is numerical (int)
#
# 2nd Output = df.head()
#
# Same as above, because your DataFrame was only 4 rows.
#
# 👉 head() = “first few rows dekh ke confirm karo data sahi load hua”.
#
# Exam me ALWAYS do:
#
# df.head()
# df.info()
# df.describe()
#
# 3rd Output = selecting 2 columns
# name  salary
# A     50000
# B     80000
# C     60000
# D     90000
#
#
# 👉 Yeh kaam exam me VERY frequent hota hai:
#
# Model ke input features choose karna
#
# Graph banane ke liye column select karna
#
# Salary distribution check karna
#
# Groupby karna
#
# 4th Output = filtering rows (df[df['age'] > 28])
# name  age  salary
# B     30   80000
# C     29   60000
# D     40   90000
#
#
# 👉 This is conditional filtering, one of the most important Pandas tools.
#
# Exam me iss se:
#
# low-income group
#
# high-age group
#
# employees above threshold
#
# purchases > 1
#
# category selection
#
# sab solve hota hai.