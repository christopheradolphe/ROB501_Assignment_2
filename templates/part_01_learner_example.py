import numpy as np
import matplotlib.pyplot as plt
from saddle_point import saddle_point

# Build non-smooth but noise-free test patch.
Il = np.hstack((np.ones((10, 10)), np.zeros((10, 10)))) 
Ir = np.hstack((np.zeros((10, 10)), np.ones((10, 10))))
I = np.vstack((Il, Ir))

pt = saddle_point(I)
print('Saddle point is at: (%.2f, %.2f)' % (pt[0, 0], pt[1, 0]))
plt.imshow(I, cmap = 'gray')
plt.plot(pt[0, 0], pt[1, 0], 'r+')
plt.show()