"""
===========================
Code to quickly visualise a datset
===========================
"""

import skdim
import matplotlib.pyplot as plt

dataset = skdim.datasets.hyperSphere(n=1000, d=3, radius=1)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(projection='3d')
ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2])
ax.axis('equal')
ax.set_title("First three dimensions of dataset")
plt.show()
