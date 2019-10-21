# # transpose matrix
# from numpy import array
# A = array([[1, 2], [3, 4], [5, 6]])
# print(A)
# C = A.T
# print(C)
##################################################################################################################
# # invert matrix
# from numpy import array
# from numpy.linalg import inv
# # define matrix
# A = array([[1.0, 2.0], [3.0, 4.0]])
# print("A)
# # invert matrix
# B = inv(A)
# print(B)
# # Dot product of  A and B
# I = A.dot(B)
# print(I)
##################################################################################################################
# # trace
# from numpy import array
# from numpy import trace
# A = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(A)
# B = trace(A)
# print("Trace: ",B)
##################################################################################################################
# #Matrix arithmatic
# import numpy as np
# x = np.array([1,5,2])
# y = np.array([7,4,1])
# print(x + y)
# print(x-y)
# print(x/y)
# print(x%y)
# print(np.dot(x,y))
# print(x*y)
##################################################################################################################
# #matrix multiplication using np.array - subclass of array.
# import numpy as np
# x = np.array( ((2,3), (3, 5)) )
# y = np.array( ((1,2), (5, -1)) )
# print(x)
# print(y)
# print(x * y)
# print("******************************")
# x = np.matrix( ((2,3), (3, 5)) )
# y = np.matrix( ((1,2), (5, -1)) )
# print(x * y)
##################################################################################################################
import numpy as np
NumPersons = np.array([[100, 175, 210], [90, 160, 150], [200, 50, 100], [120, 0, 310]])
cost_per_unit = np.array([2.98,3.90,1.99])
price = np.dot(NumPersons,cost_per_unit)
print(price)