import numpy as np
import matplotlib.pyplot as plt
from pose_estimate_nls import pose_estimate_nls, hpose_from_epose
from dcm_from_rpy import dcm_from_rpy

# Camera intrinsics matrix - known.
K = np.array([[564.9, 0, 337.3], [0, 564.3, 226.5], [0, 0, 1]])

# Load landmark points (3D - ground truth).
Wpts = np.load('../data/world_pts.npy')

# Load initial guess for camera pose.
Twc_guess = np.load('../data/camera_pose_guess_01.npy')

# Load detected cross-junctions (2D - in image).
Ipts = np.load('../data/cross_junctions_01.npy')

Twc = pose_estimate_nls(K, Twc_guess, Ipts, Wpts)
print(Twc)

# Twc =
#     0.9159   -0.3804    0.1282    0.0932
#     0.3827    0.9239    0.0074   -0.0082
#    -0.1212    0.0423    0.9917   -1.0947
#          0         0         0    1.0000

# Plot the pose to check!
C = dcm_from_rpy(np.array([-np.pi/2, 0, 0]))
Wpts = C@Wpts # Rotate upright to visualize.
Twc = hpose_from_epose(np.array([0, 0, 0, -np.pi/2, 0, 0]))@Twc

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.scatter3D(Wpts[0, :], Wpts[1, :], Wpts[2, :], 'o', c = 'r')
ax.scatter3D(Twc[0, 3], Twc[1, 3], Twc[2, 3], 'o', c = 'b')
s = np.hstack((Twc[0:3, None, 3], Twc[0:3, None, 3], Twc[0:3, None, 3]))
e = 0.3*Twc[0:3, 0:3] + s
ax.plot3D([s[0, 0], e[0, 0]], [s[1, 0], e[1, 0]], [s[2, 0], e[2, 0]], 'r')
ax.plot3D([s[0, 1], e[0, 1]], [s[1, 1], e[1, 1]], [s[2, 1], e[2, 1]], 'g')
ax.plot3D([s[0, 2], e[0, 2]], [s[1, 2], e[1, 2]], [s[2, 2], e[2, 2]], 'b')
ax.auto_scale_xyz([-0.3, 0.7], [-1.5, 0.1], [-0.5, 0.5])
plt.show()