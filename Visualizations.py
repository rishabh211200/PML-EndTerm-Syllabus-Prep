#### We should create some big data ##################

import pandas as pd
import numpy as np

np.random.seed(42)

n = 50

df = pd.DataFrame({
    'age': np.random.randint(22, 60, n),
    'salary': np.random.randint(30000, 120000, n),
    'experience': np.random.randint(0, 20, n),
    'city': np.random.choice(['Delhi', 'Mumbai', 'Bangalore', 'Chennai'], n),
    'department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing'], n)
})

# Add a binary purchase variable (for logistic regression)
df['purchased'] = (df['salary'] > 70000).astype(int)

# Insert some missing values intentionally
df.loc[np.random.choice(df.index, 5), 'salary'] = np.nan
df.loc[np.random.choice(df.index, 3), 'city'] = np.nan

# Add one outlier in salary
df.loc[5, 'salary'] = 500000   # Outlier

print(df.head())



#Step A : Data Cleaning

print("Missing values:")
print(df.isnull().sum())


## Output
# Missing values:
# age           0
# salary        5
# experience    0
# city          3
# department    0
# purchased     0
# dtype: int64
#
# ⭐ STEP 1 — PROFESSIONAL / EXAM-STYLE INTERPRETATION
# (THIS is directly worth 5 marks)
# ✔ 1. salary has 5 missing values
# Bahut zyada (50 rows me se 5 missing = 10%).
# Salary numeric hai → mean ya median se fill karna hoga.
# Lekin salary me ek extreme outlier (500,000) bhi hai, toh median better.
# 💡 Exam line
#
# “Salary column contains 5 missing values. Since salary has outliers (e.g., 500000), median imputation is more appropriate than mean.”
# ✔ 2. city has 3 missing values
# Categorical column
# Mode se fill karte hain
# Exam me categorical ALWAYS mode
# 💡 Exam line:
# “City column has 3 missing values. Categorical NaNs were imputed using mode.”
# ✔ 3. age, experience, department, purchased have NO missing
# No action needed
# Safe for modeling
# age + experience → ML-compatible immediately
# 💡 Exam line:
# “Age, experience, department and purchased columns contain no missing values.”
# ✔ 4. Overall dataset condition:
# Numeric missing: salary → 5
# Categorical missing: city → 3
# Outlier: salary (500000)
# This is exactly the kind of cleanup ML pipeline me ki jati hai.



print(df['salary'].median())

print(df['city'].mode()[0])

print(df['salary'].describe())



# Fill missing salary values
df['salary'] = df['salary'].fillna(df['salary'].median())

# Fill missing city values
df['city'] = df['city'].fillna(df['city'].mode()[0])

# Remove salary outlier
df = df[df['salary'] < 200000]

# Check summary
print(df['salary'].describe())

#Scatter Plot

import matplotlib.pyplot as plt

plt.scatter(df['age'], df['salary'])
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Age vs Salary')
plt.show()
#Output - AgeVsSalary.png

#Comments:
#
# ⭐ PROFESSIONAL / EXAM-STYLE INTERPRETATION
#
# for your scatter:
#
# ✔ 1. No clear linear relationship
#
# “The scatter plot does not show any clear linear relationship between age and salary.”
#
# ✔ 2. High variability
#
# “Salary values vary widely across almost all age groups, indicating high variance.”
#
# ✔ 3. Weak or no correlation
#
# “There appears to be weak or no correlation between age and salary.”
#
# ✔ 4. Possible slight pattern (optional)
#
# If you want to mention something subtle:
#
# “A slight upward trend may exist for some individuals, but overall the relationship is not consistent.”
#
# ✔ 5. Conclusion
#
# “Age alone does not seem to be a strong predictor of salary in this dataset.”
#
# ⭐ WHY THIS HAPPENS?
#
# Because salary distribution depends on:
#
# Experience
#
# Education
#
# Department
#
# Company level
#
# City cost of living
#
# Age alone is not enough → hence the weak pattern.
#
# ⭐ EXAM MARKS MILNE WALE POINTS (SUPER IMPORTANT)
#
# Agar tum EXACT yeh likh doge:
#
# “The scatter plot does not show a strong trend. Salary is highly spread across age groups, indicating weak or no correlation.”


plt.hist(df['salary'], bins=10)
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.title('Salary Distribution')
plt.show()
#Output - SalaryDistributions.png

#Comments:

# 1. Most frequent salary range (mode region):
#
# “The most common salary range is ₹70,000–₹80,000, with around 12 occurrences.”
#
# 2. Distribution shape (very important):
#
# “The overall distribution appears fairly spread out with moderate right-skewness.”
#
# (Why slight right skew? → You have fewer people earning >1 lakh.)
#
# 3. Salary diversity:
#
# “Salaries range from around ₹30,000 to ₹1,20,000, indicating high variance.”
#
# 4. Outlier removed successfully:
#
# “There is no extreme outlier present after cleaning (500k was removed earlier).”




#BoxPlot
plt.boxplot(df['salary'])
plt.title('Boxplot of Salary')
plt.ylabel('Salary')
plt.show()
