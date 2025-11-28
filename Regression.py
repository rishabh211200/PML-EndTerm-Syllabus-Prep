#### Using same data used while visualization, after cleaning an all ##################

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


#****************************  SKLearn  Linear Regression ************************************

from sklearn.linear_model import LinearRegression

# Features and target
X = df[['age', 'experience']]     # inputs
y = df['salary']                  # target

# Create model
model = LinearRegression()

# Fit model
model.fit(X, y)
print('************************************************************************')
# Print learned parameters
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

#
# ⭐ YOUR OUTPUT
# Coefficients: [-325.89800111  570.28037105]
# Intercept: 80706.60073599851
#
#
# Let’s break it down.
#
# ⭐ PART 1 — What is the Linear Regression formula here?
#
# Given 2 features:
#
# X₁ = age
#
# X₂ = experience
#
# Your model learned this equation:
#
# salary
# =
# −
# 325.89
# ⋅
# age
# +
# 570.28
# ⋅
# experience
# +
# 80706.60
# salary=−325.89⋅age+570.28⋅experience+80706.60
#
# This is your predicted salary formula.
#
# ⭐ PART 2 — Coefficient 1 → Age = –325.89
# Meaning (baby-level):
#
# Jitni age badhti hai:
#
# → Salary 325 rupees kam hoti dikh rahi hai (on average)
# → Matlab “age” salary ko aggressively increase nahi kar rahi
#
# Why this happened?
#
# Because your dataset is random —
# salary was NOT strongly correlated with age.
# Heatmap me bhi age ↔ salary = –0.11 weak negative tha.
#
# Model ne usi pattern ko learn kiya.
#
# Exam wording:
#
# “Age has a very weak negative impact on salary (coefficient ≈ –326).
# This indicates that age alone is not a strong predictor of salary in this dataset.”
#
# ⭐ PART 3 — Coefficient 2 → Experience = +570.28
# Meaning (baby-level):
#
# Har 1 year experience add karne se:
#
# → Salary approx ₹570 badhne ka pattern model ne dekha.
#
# Iska matlab experience salary ka stronger predictor hai age se
# (heatmap me experience ↔ salary also weak but still more than age)
#
# Exam wording:
#
# “Experience has a positive coefficient (~570), indicating that each extra year of experience increases expected salary by about ₹570.”
#
# ⭐ PART 4 — Intercept = ₹80,706.60
# Meaning (simple):
#
# Agar kisi bande ki:
#
# age = 0
#
# experience = 0
#
# Toh model guess karta salary ≈ ₹80,706.
#
# Real-life ye meaningful nahi hota, bas mathematically line ko start karne ka point hota hai.
#
# Exam wording:
#
# “The intercept (~80706) represents the base salary when all features are zero. It is not directly meaningful but is part of the regression line.”
#
# ⭐ PART 5 — Is the model good or bad?
# Based on coefficients:
#
# Age effect: tiny, almost zero → weak
#
# Experience effect: small → weak
#
# Salary doesn’t depend much on these two in your dataset
#
# Based on heatmap:
#
# Age ↔ salary = –0.11
# Experience ↔ salary = 0.09
# → Weak correlations detect ho rahe the
# So model is weak, which is OK for exam.
#
# Exam wording:
#
# “Both features have weak effects. Thus, the linear regression model is not a strong predictor of salary on this dataset.”
#
# ⭐ PART 6 — Your interpretation answer (copy-paste ready)
#
# “Linear regression learned the equation:
# salary = –325.89 × age + 570.28 × experience + 80706.60
#
# Age has a very small negative coefficient, showing weak negative relation with salary. Experience has a small positive coefficient (~570), indicating a mild positive impact. The intercept (~80706) represents the baseline salary.
#
# Overall the model shows weak predictive strength, which matches the correlation heatmap that showed low relationships among variables.”
#
# This is EXACTLY what examiner wants.


#****************************  SKLearn  Linear Regression ************************************

from sklearn.linear_model import LogisticRegression
df['city_code'] = df['city'].astype('category').cat.codes
df['dept_code'] = df['department'].astype('category').cat.codes

# Prepare features and target
X = df[['age', 'experience', 'salary', 'city_code', 'dept_code']]
y = df['purchased']

# Create model
log_model = LogisticRegression(max_iter=1000)

# Fit model
log_model.fit(X, y)
print('************************************************************************')
print("Coefficients:", log_model.coef_)
print("Intercept:", log_model.intercept_)

# ⭐ YOUR LOGISTIC REGRESSION OUTPUT
# Coefficients:
# [-0.31333124   0.36619007   0.00633852  -0.84693227  -0.605635 ]
# Intercept: -434.43526319
#
#
# Note:
# Features ka order exactly ye hoga:
#
# age
#
# experience
#
# salary
#
# city_code
#
# dept_code
#
# So coefficient list also same order me hai.
#
# ⭐ PART 1 — Logistic Regression kya sikhta hai?
#
# It learns a formula:
#
# #p=sigmoid(a1⋅age+a2⋅experience+a3⋅salary+a4.city_code+a5⋅dept_code+b)
#
# Yani har feature ka effect purchase probability par.
#
# ⭐ PART 2 — HOW TO READ COEFFICIENTS
#
# Simple rule:
#
# ✔ Positive coefficient → likelihood of 1 increases
#
# (“Purchased = YES” hone ka chance badhta hai)
#
# ✔ Negative coefficient → likelihood of 1 decreases
#
# (“Purchased = YES” hone ka chance kam hota hai)
#
# ⭐ NOW LET'S READ YOUR COEFFICIENTS ONE-BY-ONE:
# ✔ 1) age coefficient = –0.31 (NEGATIVE)
#
# Meaning:
#
# As age increases → purchase chance slightly decreases
#
# Very small effect → almost no influence
#
# EXAM LINE:
#
# “Age has a weak negative impact on purchase probability.”
#
# ✔ 2) experience = +0.366 (POSITIVE)
#
# Meaning:
#
# More experience → higher chance of purchase
#
# But effect still mild
#
# EXAM LINE:
#
# “Experience has a mild positive influence on purchasing behavior.”
#
# ✔ 3) salary coefficient = +0.0063 (SMALL POSITIVE)
#
# Salary ka effect bahut chhota lag raha hai, because value salary unit me directly enters model.
#
# BUT after scaling, even small numbers can be meaningful.
#
# EXAM LINE:
#
# “Salary shows a small positive relationship with purchasing likelihood.”
#
# (Heatmap me purchased ↔ salary = strong positive tha, but logistic regression unscaled features me dull ho sakta hai.)
#
# ✔ 4) city_code = –0.8469 (STRONG NEGATIVE)
#
# This is the strongest coefficient in the model.
#
# Meaning:
#
# City ka type purchase decision ko STRONGLY affect karta hai
#
# Kuch city groups me purchase probability low hai
#
# City category plays major role in customer behavior
#
# EXAM LINE:
#
# “City_code has the strongest negative coefficient, indicating location heavily influences purchasing likelihood.”
#
# ✔ 5) dept_code = –0.6056 (MODERATE NEGATIVE)
#
# Meaning:
#
# Some departments are less likely to purchase
#
# Department matters moderately
#
# EXAM LINE:
#
# “Department_code shows a moderate negative impact on purchasing probability.”
#
# ⭐ PART 3 — WHAT ABOUT INTERCEPT?
# Intercept = -434.43
#
#
# Simple meaning:
#
# Base probability = VERY LOW
#
# Model predicts purchase = mainly if features raise the linear score significantly
#
# Exam line:
#
# “The intercept is large and negative, showing low baseline purchase probability when all features are zero.”
#
# ⭐ PART 4 — FULL EXAM INTERPRETATION (copy-paste ready)
#
# “Logistic regression shows that experience and salary have mild positive effects on purchase probability, while age has a weak negative impact. City_code has the strongest negative coefficient (–0.84), indicating location plays the most significant role in purchasing behavior. Department_code also reduces purchase likelihood moderately. The intercept (–434) indicates low baseline purchase probability. Thus, purchasing behavior is mostly influenced by city and department.”
#
# This = full marks guaranteed.


#**************************************************************************************************************
#Confusion Matrix + Accuracy
from sklearn.metrics import accuracy_score, confusion_matrix

# Predict on the same dataset (for now)
y_pred = log_model.predict(X)
print('*************************************************************************')
print("Accuracy:", accuracy_score(y, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y, y_pred))


#Output:

# ⭐ PART 1 — CONFUSION MATRIX (baby-level samajh)
#
# Your matrix:
#
# [[23  0]
#  [ 0 26]]
#
#
# Yeh 2×2 table hota hai:
#
# 	Predicted 0	Predicted 1
# Actual 0	TN = 23	FP = 0
# Actual 1	FN = 0	TP = 26
#
# Let's decode:
#
# ⭐ TN = 23
#
# Actual 0 → Predicted 0
# ✔ Model ne “NO PURCHASE” sahi bola
# ✔ 23 log accurate
#
# ⭐ FP = 0
#
# Actual 0 → Predicted 1
# ❌ Model ne “purchase karega” bola but actually 0 tha
# ✔ ZERO mistakes here
#
# ⭐ FN = 0
#
# Actual 1 → Predicted 0
# ❌ Model ne “purchase nahi karega” bola but actually 1 tha
# ✔ ZERO mistakes here
#
# ⭐ TP = 26
#
# Actual 1 → Predicted 1
# ✔ Model ne “purchase yes” sahi bola
#
# ⭐ PART 2 — Accuracy = 1.0 (100%)
#
# Meaning:
#
# Model ne 1 bhi case galat nahi kiya
#
# PERFECT prediction
#
# But BUT BUT…
#
# 👉 Yeh exam me explain karna zaroori hai:
#
# “Such perfect accuracy usually indicates overfitting, because prediction was done on the same data used for training.”
#
# Agar train-test split karoge → accuracy kam hogi → realistic.
#
# ⭐ PART 3 — Exam-Ready Interpretation (copy-paste)
#
# “The confusion matrix shows 23 true negatives and 26 true positives, with zero false positives and zero false negatives. This results in an accuracy of 100%. Since predictions were made on the training data itself, this high accuracy likely reflects overfitting. A train-test split is required for realistic performance measurement.”
#
# Yeh lines = Full marks.


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit model on training data
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Predict on test data
y_pred_test = log_model.predict(X_test)
print('***********************************************************************')
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))

#output:
#
# Aur ab main tumhe CONFUSION MATRIX + TEST ACCURACY ko ekdum beginner → advanced → exam topper style me samjha deta hoon.
#
# ⭐ YOUR TEST RESULTS
# Test Accuracy: 0.9
# Confusion Matrix:
# [[5 1]
#  [0 4]]
#
#
# Let’s decode EVERYTHING step-by-step.
#
# ⭐ PART 1 — CONFUSION MATRIX ko samajhna (baby level)
#
# Confusion matrix:
#
# 	Predicted 0	Predicted 1
# Actual 0	5	1
# Actual 1	0	4
#
# Now decode:
#
# ✔ True Negatives (TN) = 5
#
# Actual 0 → Predicted 0
# Model ne NO PURCHASE sahi bola
# ✔ 5 customers accurately identified as “not purchasing”
#
# ✔ False Positives (FP) = 1
#
# Actual 0 → Predicted 1
# Model ne bola customer buy karega, but actually nahi kiya
# ❌ 1 mistake here
# (ye “over-confident positive prediction” hota hai)
#
# ✔ False Negatives (FN) = 0
#
# Actual 1 → Predicted 0
# Model ne customer ko “not purchase” bola, but he actually purchased
# ❌ yeh dangerous mistake hoti hai
# ✔ Par yaha 0 mistakes → GOOD
#
# ✔ True Positives (TP) = 4
#
# Actual 1 → Predicted 1
# 4 customers correctly predicted as “purchased”
#
# ⭐ PART 2 — Accuracy = 0.90 (90%)
#
# Meaning:
#
# Total test cases = 10
#
# Correct predictions = 9
#
# Wrong predictions = only 1
#
# Yeh bohot achha result hai exam ke liye.
#
# Training accuracy 100% thi
# Test accuracy 90% →
# Model is not overfitting much.
#
# Professor loves this point.
#
# ⭐ PART 3 — FULL EXAM-WORTHY INTERPRETATION (copy-paste ready)
#
# “On the test data, the logistic regression model achieved 90% accuracy. The confusion matrix shows 5 true negatives, 4 true positives, 1 false positive, and zero false negatives. The absence of false negatives indicates the model correctly identifies all actual purchasers. Since test accuracy (90%) is lower than training accuracy (100%), the model generalizes well with mild overfitting.”
#
# Yeh EXACT answer examiner love karta hai.
#
# ⭐ PART 4 — Practical Insight Jo Professor dekhna chahta hai
# ✔ No false negatives
#
# → Model never misses a customer who will purchase
# → Business-friendly result
# → Good sensitivity
#
# ✔ Only 1 false positive
#
# → Model ne ek galat khush banda declare kiya
# → Acceptable
#
# ✔ Balanced performance
#
# → Good generalization
# → No high overfitting
#
# ⭐ PART 5 — What next? (Clustering time!)
#
# Now that Linear + Logistic complete:


