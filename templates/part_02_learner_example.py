import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from cross_junctions import cross_junctions

# Load the world points.
Wpts = np.load('../data/world_pts.npy')

# Load the example target image.
I = imread("../images/target_01.png")

# Load the bounding polygon.
bpoly = np.load('../data/bounds_01.npy')

# Load the reference solution and compute yours.
Ipts_ref = np.load('../data/cross_junctions_01.npy')
Ipts = cross_junctions(I, bpoly, Wpts)

# Plot the points to check!
plt.imshow(I, cmap = 'gray')

plt.plot(Ipts_ref[0, :], Ipts_ref[1, :], 'o', c = 'r', markersize = 8)
for i in range(0, Ipts_ref.shape[1]):
    plt.text(Ipts_ref[0, i] - 10, Ipts_ref[1, i] - 10, str(i + 1), c = 'r')

bpoly = np.append(bpoly, bpoly[:, None, 0], axis = 1) # Close the polygon.
plt.plot(bpoly[0, :], bpoly[1, :], '-', c = 'b', linewidth = 3)
plt.plot(bpoly[0, 0], bpoly[1, 0], 'x', c = 'b', markersize = 9)
plt.text(bpoly[0, 0] - 40, bpoly[1, 0] - 10, "Upper Left", c = 'b')

plt.plot(Ipts[0, :], Ipts[1, :], 'o', c = 'g',)
plt.show()
