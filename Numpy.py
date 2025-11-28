# #This includes:
#
# #✔ NUMPY = fast maths + fast arrays
#
# # Tu koi bhi ML model banane jaayega — peeche hidden NumPy hi kaam karta hai.
# # Imagine Excel sheet me bahut numbers hain, tum unpe:
# # Add
# # Subtract
# # Multiply
# # Average
# # Square
# # Vector operations
# # Sab fast chahte ho na?
# # Yeh NumPy deta hai.
#
# # ✔ 1. Create array
#
# import numpy as np
# a = np.array([1, 2, 3, 4])
# print(a)
#
# # ✔ 2. Shape and size
#
# print(a.shape)
# print(a.size)
#
# # ✔ 3. 2D array
#
# b = np.array([[1,2,3],[4,5,6]])
# print(b)
#
# # ✔ 4. Indexing
#
# print(a[0])
# print(b[1,2])
#
# # ✔ 5. Operations (super easy)
#
# print(a + 10)
# print(a * 2)
# print(a / 2)
# print(a.mean())
# print (a.std())
#
# #1d & 2D understanding
#
# a = np.array([10, 20, 30, 40])   # shape: (4,)
# b = a.reshape(4,1)               # shape: (4,1)
# c = a.reshape(1,4)               # shape: (1,4)
# print (a)
# print (b)
# print (c)
#
#
# # ✔ 6. Useful stuff for ML
#
# x = np.arange(1,10)     # 1 se 9 tak numbers
# y = x.reshape(3,3)      # 3x3 matrix
# print(x)
# print(y)


#************************Micro Task for Numpy*****************************

import numpy as np

a = np.array([10, 20, 30, 40])
b = np.array([[1, 2, 3],
              [4, 5, 6]])

print("a =", a)
print("b =", b)
print("a mean =", a.mean())
print("a std =", a.std())
print("b shape =", b.shape)
print("b[1, 2] =", b[1,2])
