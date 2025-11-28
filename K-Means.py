# we will use real data now, pehele ki trh senseless data use nhi kregey kuki patterns smjhne hai ab hamein:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Create clean 3 clusters
cluster_0 = pd.DataFrame({
    'salary': np.random.randint(30000, 50000, 30),
    'experience': np.random.randint(0, 3, 30),
    'cluster_true': 0
})

cluster_1 = pd.DataFrame({
    'salary': np.random.randint(60000, 80000, 30),
    'experience': np.random.randint(4, 8, 30),
    'cluster_true': 1
})

cluster_2 = pd.DataFrame({
    'salary': np.random.randint(100000, 150000, 30),
    'experience': np.random.randint(9, 15, 30),
    'cluster_true': 2
})

# Combine into one dataset
df_clusters = pd.concat([cluster_0, cluster_1, cluster_2], ignore_index=True)

# Plot the true clusters
plt.figure(figsize=(7,5))
plt.scatter(df_clusters['salary'], df_clusters['experience'], c=df_clusters['cluster_true'], cmap='viridis')
plt.xlabel('Salary')
plt.ylabel('Experience')
plt.title('TRUE Clusters (Meaningful)')
plt.show()

print(df_clusters.head())


#****************************Apply K-Means on data*****************************************

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

Xc = df_clusters[['salary', 'experience']]

kmeans = KMeans(n_clusters=3, random_state=42)
df_clusters['cluster_kmeans'] = kmeans.fit_predict(Xc)

print("Cluster Centers:\n", kmeans.cluster_centers_)

plt.figure(figsize=(7,5))
plt.scatter(df_clusters['salary'], df_clusters['experience'], c=df_clusters['cluster_kmeans'], cmap='viridis')
plt.xlabel('Salary')
plt.ylabel('Experience')
plt.title('KMeans Clusters (Learned)')
plt.show()


#Output & Comments

# ⭐ YOUR CLUSTER CENTERS (Decoded)
# Cluster 0 : [67923 , 5.60]
# Cluster 1 : [125731 , 11.56]
# Cluster 2 : [38754 , 0.97]
#
#
# Let’s interpret them:
#
# ⭐ CLUSTER 1 — ₹1.25 lakh salary, 11.5 yrs experience
#
# → Senior / High Salary Group
#
# ✔ High salary group
# ✔ High experience group
# ✔ Most experienced people
# ✔ Most likely highest performers / senior employees
#
# Exam line:
#
# “Cluster 1 shows the highest salary (~125k) and highest experience (~12 yrs), representing senior-level professionals.”
#
# ⭐ CLUSTER 0 — ₹67k salary, 5.6 yrs experience
#
# → Mid-Level Group
#
# ✔ Salary ~70k
# ✔ Experience ~5.6 yrs
# ✔ These are mid-career, stable employees
# ✔ Not too junior, not too senior
#
# Exam line:
#
# “Cluster 0 represents mid-level professionals with moderate salary and experience.”
#
# ⭐ CLUSTER 2 — ₹38k salary, ~1 yr experience
#
# → Entry-Level / Beginners
#
# ✔ Low salary
# ✔ Very low experience
# ✔ Freshers or early-career employees
# ✔ Just joined industry
#
# Exam line:
#
# “Cluster 2 contains low salary and low experience employees, representing entry-level workers.”
#
# ⭐ SUPER IMPORTANT EXAM POINT
#
# You already said it:
#
# “They seem matching, and cluster is same as I had given you.”
#
# YES — that’s EXACTLY what KMeans should do on meaningful data.
#
# Tumne perfect pattern pakda:
#
# 👉 Real clusters (ground truth)
#
# = KMeans clusters (machine-learned)
#
# 👉 This proves
#
# Data me natural structure tha
#
# KMeans ne use correctly detect kiya
#
# YOU NOW KNOW how real clusters look
#
# ⭐ EXAM-READY INTERPRETATION (Copy-Paste)
#
# “KMeans with 3 clusters produced well-separated groups.
# One cluster contains low salary and low experience employees (entry-level).
# The second cluster contains mid-salary and mid-experience individuals (mid-level).
# The third cluster contains high salary and high experience employees (senior professionals).
# The cluster centers closely match the true underlying groups, indicating effective clustering.”
#
# This = full marks.