from matplotlib import pyplot as plt
import numpy as np



data = []
file_in = open('output.txt', 'r')
for y in file_in.readlines():
    if y[0].isdigit() :
        data.append(float(y))

N = 10000
loss1 = np.convolve(data, np.ones(N)/N, mode='valid')

#plt.plot(data,color = 'lightsalmon', linewidth=1)
plt.plot(loss1,color = 'teal', linewidth=1)
plt.ylabel('loss')
plt.show()