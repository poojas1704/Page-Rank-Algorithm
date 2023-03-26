#!/usr/bin/env python
# coding: utf-8

# Part 1: Trying Page Rank Algorithm on toy matrix

# In[91]:


import numpy as np


# In[92]:


from numpy.linalg import norm


# In[93]:


import pandas as pd


# 1.2 Creating an Adjacency Matrix (Z) # Need to generalize this 

# In[94]:


z = np.array([[1,0,2,0,4,3], [3,0,1,1,0,0], [2,0,4,0,1,0], [0,0,1,0,0,1], [8,0,3,0,5,2], [0,0,0,0,0,0]])
print(z)
print(z.shape)


# 1.3 Modifying the Adjacency Matrix

# 1. Set diagonals to zero 

# In[95]:


np.fill_diagonal(z, 0)

# for i in range(0,6):
#     for j in range(0,6):
#         if i==j:
#             z[i][j]=0
print(z)
print(z.shape)


# 2. Normalize the columns - gives matrix H

# In[96]:


# Creating an array for storing the sum of each column 
hsum = [0] * 6
for i in range(0,6):
    for j in range(0,6):
        hsum[j] = hsum[j] + z[i][j]
print(hsum)


# In[97]:


# Normalizing each column
h = np.zeros((6,6))
for i in range(6):
    for j in range(6):
        if hsum[i] != 0:
            h[j][i] = z[j][i]/hsum[i]
print(h)
print(h.shape)


# 1.4 Identifying the Dangling Nodes

# In[98]:


# If all elements in a column are zero, then corresponding value in dangling node will be 1 
d = np.array([0,0,0,0,0,0])
for i in range(6):
    count = 0
    for j in range(6):
        if h[j][i]==0:
            count+=1
    if count == 6:
        d[i] = 1
print(d)
print(d.shape)
        


# 1.5 Calculating the Influence Vector

# In[99]:


# Creating Article Vector
a = np.array([[3,2,5,1,2,1]])
a=a.T

# Normalizing Article Vector
asum = 0
for i in range(6):
    asum = asum + a[i][0]
a = a/asum
print(a)
print(a.shape)


# In[100]:


# Creating Initial Vector
initial_vector = np.array([[1/6],[1/6],[1/6],[1/6],[1/6],[1/6]])
print(initial_vector)
print(initial_vector.shape)


# In[101]:


# Calculating influence vector: Method 1

# Calculating P

# Creating H_edited



# h_edited = h
# for i in range(0,6):
#     count = 0
#     for j in range(0,6):
#         if h[j][i]==0:
#             count+=1
#     if count == 6:
#         for j in range(6):
#             h_edited[j][i] = a[j]
# #print(h_edited)
   
# et = np.array([1,1,1,1,1,1])
    
# p = 0.85*h_edited + 0.15*(np.dot(et,a))

# print(p)


# In[102]:


# Calculating Influence Vector: Method 2

inf_vec = np.array([0,0,0,0,0,0]) # Creating influence vector of shape (1,6), where all elements = 0
pi_initial = initial_vector
cond = 1 # Providing initial condition to enter while loop
count = 0 # count variable used to count the number of iterations
while cond >= 0.00001:
    inf_vec = 0.85*np.dot(h,pi_initial) + np.dot(a,[0.85*np.dot(d,pi_initial)+0.15])
    arr = np.subtract(inf_vec,pi_initial)
    cond = np.linalg.norm(arr, ord=1)
    pi_initial=inf_vec
    count+=1
print(count)
print(inf_vec)


# 1.6 Calculating Eigenfactor (EFi)

# In[103]:


# Calculating Eigenfactor

EF = np.dot(h, inf_vec)
#print(EF)
EF_sum =0

# Calculation sum of EF matrix elements for normalization
for i in range(6):
    EF_sum = EF_sum + EF[i]
#print(EF_sum)

# Normalizing EF matrix and multiplying by 100
EF = (EF/EF_sum)*100

print(EF)


# Part 2: Trying Page Rank Algorithm on actual data

# In[104]:


# Calculating start time to evaluate time taken to run program
import time
start = time.time()

# Load data from the file 
file = pd.read_csv("links.txt", sep=',', names=['Journal1', 'Journal2', 'Citations'])
print(file.head(10))


# 1.2 Creating an Adjacency Matrix (Z) 

# In[105]:


n=10748
adjacency_matrix = np.zeros((n,n))

#Iterating over each row and inserting citation value in corresponding journal pair
for index, row in file.iterrows():
    i=int(row['Journal1'])
    j=int(row['Journal2'])
    adjacency_matrix[j][i] = row['Citations']
        
#print(adjacency_matrix.shape())


# 1.3 Modifying the Adjacency Matrix

# Set diagonals to zero

# In[106]:


np.fill_diagonal(adjacency_matrix, 0)


# In[107]:


print(adjacency_matrix)


# Normalize the columns

# In[108]:


# Creating a matrix which stores the sum of each column
matrix_hsum = [0] * n
for i in range(n):
    for j in range(n):
        matrix_hsum[j] = matrix_hsum[j] + adjacency_matrix[i][j]
#print(matrix_hsum)


# In[109]:


# Normalizing each column
matrix_h = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        if matrix_hsum[i] != 0:
            matrix_h[j][i] = adjacency_matrix[j][i]/matrix_hsum[i]
print(matrix_h)


# 1.4 Identifying the Dangling Nodes

# In[110]:


# If all elements in a column are zero, then corresponding value in dangling node will be 1 
dangling_node = np.zeros((n))
for i in range(n):
    count = 0
    for j in range(n):
        if matrix_h[j][i]==0:
            count+=1
    if count == n:
        dangling_node[i] = 1
print(dangling_node)
print(dangling_node.shape)


# 1.5 Calculating the Influence Vector

# In[111]:


# Creating Article Vector
article_matrix = np.full((1, n), 1)
article_matrix=article_matrix.T

# Normalizing Article Vector
article_matrix_sum = 0

for i in range(n):
    article_matrix_sum = article_matrix_sum + article_matrix[i][0]
article_matrix = article_matrix/article_matrix_sum
print(article_matrix)
print(article_matrix.shape)


# In[112]:


# Creating initial vector
initial_vector_matrix = np.full((n, 1), 1/n) # Creating initial vector of shape (n,1), where all elements = 1/n
print(initial_vector_matrix)
print(initial_vector_matrix.shape)


# In[113]:


# Calculating Influence Vector

influence_vector = np.full((1, n), 0) # Creating influence vector of shape (1,n), where all elements = 0
pi_initial_matrix = initial_vector_matrix 
condition = 1 # Providing initial condition to enter while loop
print(condition)
c = 0 # c variable used to count the number of iterations

while condition >= 0.00001:
    influence_vector = 0.85*np.dot(matrix_h,pi_initial_matrix) + np.dot(article_matrix,[0.85*np.dot(dangling_node,pi_initial_matrix)+0.15])
    array_temp = np.subtract(influence_vector,pi_initial_matrix)
    condition = np.linalg.norm(array_temp, ord=1)
    pi_initial_matrix=influence_vector
    c+=1
    
print(c)
print(influence_vector)


# 1.6 Calculating Eigenfactor (EFi)

# In[114]:


# Calculating Eigenfactor

Eigenfactor = np.dot(matrix_h, influence_vector)
Eigenfactor_sum =0

# Calculation sum of EF matrix elements for normalization
for i in range(n):
    Eigenfactor_sum = Eigenfactor_sum + Eigenfactor[i]

# Normalizing EF matrix and multiplying by 100
Eigenfactor = (Eigenfactor/Eigenfactor_sum)*100

print(Eigenfactor)


# In[115]:


# Calculating end time to evaluate time taken to run program
end = time.time()


# In[116]:


# Sorting the eigen factor array and displaying top 20 scores

print("Scores of top 20 journals are as follows:")

Eigenfactor = Eigenfactor[Eigenfactor[:, 0].argsort()][::-1][:20]
print(Eigenfactor)


# In[117]:


np.max(Eigenfactor)


# In[118]:


# Evaluating time taken for the program to run
print("Time it took to run the code on real network")
print(end - start)


# In[119]:


print("Number of iterations it took to get to the answer:")
print(c)

