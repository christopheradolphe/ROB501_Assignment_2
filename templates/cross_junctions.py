import numpy as np
from numpy.linalg import inv, lstsq
from scipy.linalg import null_space
from scipy.ndimage.filters import *
from matplotlib.path import Path
from matplotlib import pyplot as plt

def create_patch(image, point, patch_size):
    """
    Find patch size around a point.

    Given an image, a valid point within the image and a patch size,
    return the patch that has a window on x and y of patch size around the centred point.

    Parameters:
    ----------- 
    image        - Single-band (greyscale) image as np.array (e.g., uint8, float).
    point        - 2x1 np array (x,y) point corresponding to centre of patch.
    patch_size   - size of patch around the point.

    Returns:
    --------
    patch - Single-band (greyscale) image as np.array.
    """

    # Find patch coordinates ensuring they are within bound of image
    patch_y1 = int(max(point[1] - patch_size, 0))
    patch_y2 = int(min(point[1] + patch_size, image.shape[0] - 1))
    patch_x1 = int(max(point[0] - patch_size, 0))
    patch_x2 = int(min(point[0] + patch_size, image.shape[1] - 1))

    return image[patch_y1:patch_y2, patch_x1:patch_x2]

def saddle_point_copy(I):
    # Same function as function in part1 just copied over and renamed
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
    alpha = x[0]
    beta = x[1]
    gamma = x[2]
    delta = x[3]
    epsilon = x[4]
    zeta = x[5]

    
    # 2. Find the coordinates of the Saddle Point
    # Use formula from Luccesse Paper
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

def refine_saddle_point(image, point, patch_size):
    """
    Find saddle point from a patch.

    Given an image, a valid point within the image and a patch size,
    find the saddle point in a window of patch size around the point.

    Parameters:
    ----------- 
    image        - Single-band (greyscale) image as np.array (e.g., uint8, float).
    point        - 2x1 np array (x,y) point corresponding to estimated cross junction point.
    patch_size   - size of patch to look for saddle point around the point.

    Returns:
    --------
    saddle_point - 2x1 np array (x,y) of saddle point in image.
    """
    
    patch = create_patch(image, point, patch_size)
    patch = gaussian_filter(patch,4) # Chosen standard deviation from Luccesse paper
    # Local Coordinates
    local_coord = saddle_point_copy(patch)

    # Adjust to global coordinates
    global_x = int(max(point[0] - patch_size, 0)) + local_coord[0]
    global_y = int(max(point[1] - patch_size, 0)) + local_coord[1]

    return np.array([global_x, global_y])

def dlt_homography(I1pts, I2pts):
    """
    Find perspective Homography between two images.

    Given 4 points from 2 separate images, compute the perspective homography
    (warp) between these points using the DLT algorithm.

    Parameters:
    ----------- 
    I1pts  - 2x4 np.array of points from Image 1 (each column is x, y).
    I2pts  - 2x4 np.array of points from Image 2 (in 1-to-1 correspondence).

    Returns:
    --------
    H  - 3x3 np.array of perspective homography (matrix map) between image coordinates.
    A  - 8x9 np.array of DLT matrix used to determine homography.
    """

    # Create variables for point location coordinates
    x = I1pts[0, :]
    y = I1pts[1, :]
    u = I2pts[0, :]
    v = I2pts[1, :]

    # Create empty DLT Matrix
    A = np.empty((8, 9))

    # Populate the DLT Matrix
    # Add two equations to the DLT matrix for each of the four points
    for point in range(u.shape[0]):
        A[2 * point] = [-x[point], -y[point], -1,  0,    0,    0,   u[point]*x[point],  u[point]*y[point],  u[point]]
        A[2 * point + 1] = [ 0,     0,    0, -x[point], -y[point], -1,  v[point]*x[point],  v[point]*y[point],  v[point]]

    # Homography Matrix
    # H is the nullspace of matrix A
    null = null_space(A)[:,0]
    H = null.reshape((3,3))

    # Normalize DLT Matrix so bottom right entry is 1
    H = H / H[2,2]

    return H, A


def cross_junctions(I, bpoly, Wpts):
    """
    Find cross-junctions in image with subpixel accuracy.

    The function locates a series of cross-junction points on a planar 
    calibration target, where the target is bounded in the image by the 
    specified quadrilateral. The number of cross-junctions identified 
    should be equal to the number of world points.

    Note also that the world and image points must be in *correspondence*,
    that is, the first world point should map to the first image point, etc.

    Parameters:
    -----------
    I      - Single-band (greyscale) image as np.array (e.g., uint8, float).
    bpoly  - 2x4 np.array, bounding polygon (clockwise from upper left).
    Wpts   - 3xn np.array of world points (in 3D, on calibration target).

    Returns:
    --------
    Ipts  - 2xn np.array of cross-junctions (x, y), relative to the upper
            left corner of the target. The array must contain float64 values.
    """
    # Origin: Top left cross junction
    # x-axis: Right
    # y-axis: Down
    # z-axis: Into page
    # Number of Targets: size of Wpts
    targets = Wpts.shape[1]
    Ipts = np.zeros((2,targets))

    # Steps
    # 1. Find Size of Checkerboard squares in Image
    # 2. Find corners of checkerboard given 6*8 grid
    # 3. Find the Homography Matrix to Corners of Board to bpoly
    # 4. Using Homography matrix tranform points from Wpts to image
    # 5. Check Patch size around and use saddle_points function from before

    # 1. Find Size of Checkerboard squares in Image
    # First x coordinate minus second x coordinate
    size_square = Wpts[0, 1] - Wpts[0, 0]

    # 2. Find corners of checkerboard given 6*8 grid
    top_left = Wpts[:,0] + np.array([-size_square, -size_square, 0]).T
    top_right = Wpts[:,7] + np.array([size_square, -size_square, 0]).T
    bottom_left = Wpts[:, 40] + np.array([-size_square, size_square, 0]).T
    bottom_right = Wpts[:, 47] + np.array([size_square, size_square, 0]).T

    # Need to account for extra boarder
    # Values found through interpolating values on matplotlib plot
    # x_border_size = (size of board horizontally - 9 * size of square) / 2
    # y_border_size = (size of board vertically - 7 * size of square) / 2
    x_border_size = size_square * 0.3548
    y_border_size = size_square * 0.2044

    top_left += np.array([-x_border_size, -y_border_size, 0]).T
    top_right += np.array([x_border_size, -y_border_size, 0]).T
    bottom_left += np.array([-x_border_size, y_border_size, 0]).T
    bottom_right += np.array([x_border_size, y_border_size, 0]).T

    # Define corners in clockwise direction similar to bpoly
    corners = np.array([
        [top_left[0], top_right[0], bottom_right[0], bottom_left[0]],
        [top_left[1], top_right[1], bottom_right[1], bottom_left[1]]
    ])

    # 3. Find Homography Matrix to Transform points from Wpts to Image
    H, _ = dlt_homography(corners, bpoly)

    # 4. Using Homography matrix tranform points from Wpts to image
    for i in range(targets):
        world_point = np.array([Wpts[0, i], Wpts[1, i], 1])
        image_coord = H @ world_point
        image_coord /= image_coord[2]
        # Extract x and y coordinates
        image_coord = image_coord[:2]
        Ipts[0,i] = image_coord[0]
        Ipts[1, i] = image_coord[1]
    
    # plt.imshow(I, cmap = 'gray')
    # for i in range(targets):
    #     plt.plot(Ipts[0, i], Ipts[1, i], 'r+')
    # plt.show()

    # 5. Check Patch size around and use saddle_points function from before
    for i in range(targets):
        saddle_point = refine_saddle_point(I, Ipts[:,i], 25)
        Ipts[:,i] = saddle_point[:,0]



    correct = isinstance(Ipts, np.ndarray) and \
        Ipts.dtype == np.float64 and Ipts.shape[0] == 2

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Ipts