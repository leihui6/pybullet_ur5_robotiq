import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

points = []
R = 10
for theta in range(10,90,10): # height
    for beta in range(10,360,10): # XY-plane
        r_theta, r_beta = np.deg2rad(theta), np.deg2rad(beta)
        x = R * np.sin(r_theta) * np.cos(r_beta)
        y = R * np.sin(r_theta) * np.sin(r_beta)
        z = R * np.cos(r_theta)
        points.append([x,y,z])
        
points = np.array(points)
print (points.shape)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(points[:,0],points[:,1],points[:,2])
plt.show()