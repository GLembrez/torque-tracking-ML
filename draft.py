import numpy as np
from matplotlib import pyplot as plt
import random

n = 20
M = np.zeros((n,n))
for k in range(n) :
    M[k,k] = random.uniform(0.2,1)
    for i in range(1,k) :
        M[k,k-i] = random.uniform(0,M[k,k-i+1])
    for i in range(k+1,n) :
        M[k,i] = random.uniform(0,M[k,i-1])

plt.figure()
plt.imshow(M,cmap='binary')
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.show()