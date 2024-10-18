import numpy as np
from numpy.linalg import inv, lstsq

def saddle_point(I):
    """
    Locate saddle point in an image patch.

    The function identifies the subpixel centre of a cross-junction in the
    image patch I, by fitting a hyperbolic paraboloid to the patch, and then 
    finding the critical point of that paraboloid.

    Note that the location of 'pt' is relative to (-0.5, -0.5) at the upper
    left corner of the patch, i.e., the pixels are treated as covering an 
    area of one unit square.

    Parameters:
    -----------
    I  - Single-band (greyscale) image patch as np.array (e.g., uint8, float).

    Returns:
    --------
    pt  - 2x1 np.array (float64), subpixel location of saddle point in I (x, y).
    """

    # 1. Solve the Parameters in Luccesse Equation using Linear Least squares
    # Solving Ax = b where:
    # A = [x^2, x*y, , y^2, x, y, 1]
    # x = [alpha, beta, gamma, delta, epsilon, zeta]
    # b = intensity
    
    # Create empty arrays to be populated for A and b
    num_pixels = I.shape[0]*I.shape[1]
    A = np.zeros((num_pixels, 6))
    b = np.zeros(num_pixels)

    # Populate A and b
    for x in range(I.shape[1]):
        for y in range(I.shape[0]):
            pixel_index = y * I.shape[1] + x
            A[pixel_index, 0] = x**2
            A[pixel_index, 1] = x * y
            A[pixel_index, 2] = y**2
            A[pixel_index, 3] = x
            A[pixel_index, 4] = y
            A[pixel_index, 5] = 1
            b[pixel_index] = I[y, x]
    
    # Solve least squares
    x, _, _, _ = lstsq(A,b)

    # Extract the parameters
    alpha = x[0]
    beta = x[1]
    gamma = x[2]
    delta = x[3]
    epsilon = x[4]
    zeta = x[5]

    
    # 2. Find the coordinates of the Saddle Point
    # Use the formula from Luccesse Paper
    saddle_matrix = inv(
                        np.array([[2* alpha, beta],
                                    [beta, 2 * gamma]])
                        )
    saddle_vector = np.array([delta, epsilon])
    pt = - saddle_matrix @ saddle_vector
    pt = pt.reshape((2,1))

    correct = isinstance(pt, np.ndarray) and \
        pt.dtype == np.float64 and pt.shape == (2, 1)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return pt