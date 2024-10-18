import numpy as np
from numpy.linalg import inv

def find_jacobian(K, Twc, Wpt):
    """
    Determine the Jacobian for NLS camera pose optimization.

    The function computes the Jacobian of an image plane point with respect
    to the current camera pose estimate, given a landmark point. The 
    projection model is the simple pinhole model.

    Note that the homogeneous transformation matrix provided defines the
    transformation from the *camera frame* to the *world frame* (to 
    project into the image, you would need to invert this matrix).

    Parameters:
    -----------
    K    - 3x3 np.array, camera intrinsic calibration matrix.
    Twc  - 4x4 np.array, homogenous pose matrix, current guess for camera pose. 
    Wpt  - 3x1 world point on calibration target (one of n).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian matrix (columns are tx, ty, tz, r, p, q).
         The Jacobian must contain float64 values.
    """
    # Initial Guess: +/-10 degrees and 20cm

    # Jacobian: 2x6
    # 1st row: x error
    # 2nd row: y error

    # Extract rotation matrix and translation vector from Twc
    # TWC is pose of camera with respect to target
    # C_wc is rotation of the camera frame with respect to target
    C_wc = Twc[:3,:3]
    # t: translation of camera with respect to target
    t = Twc[:3,3].reshape((3,1))
    Wpt = Wpt.reshape((3, 1))

    # Transform Target (World) Point to Camera Frame
    Cpt = inv(C_wc) @ (Wpt-t)

    # Project point onto image plane (wrt to camera)
    x_proj_hom = K @ Cpt
    x_proj = x_proj_hom / x_proj_hom[2]

    # Initialize the Jacobian
    J = np.zeros((2, 6))
    
    # Translation Components of the Jacobian
    # 3 parts to calulation of x_projected
    # 1. Camera Coordinates: pc = Ccw(pw - t)
    # 2. Projection: x_proj_hom = K * pc
    # 3. Normalization: x_proj = x_proj_hom / x_proj_hom[2]
    # Can now calculate the derivative of translation components tx, ty, tz in Twc by taking chain rule


    # Pre Compute inverse of Rotational Matrix for Camera Coordinates Partial Derivative
    inv_C_wc = inv(C_wc)

    # Pre Compute Partial Derivative of Projection Step (used in rotational and translational)
    dxprojhom_dpc = K

    # Pre compute partial derivatives of normalization step (used in rotational and translational)
    x = x_proj_hom[0,0]
    y = x_proj_hom[1,0]
    z = x_proj_hom[2,0]
    dxproj_dxprojhom_matrix = np.array([
        [1/z, 0, -x/z**2],
        [0, 1/z, -y/z**2]
    ])

    for i in range(3):
        e_i = np.zeros((3, 1))
        e_i[i, 0] = 1
        # 1. Partial derivative of pc wrt ti (dpc/dt_i)
        dpc_dti = - inv_C_wc @ e_i
        # 2. Partial derivative of x_proj_hom to pc already computed (K)
        # 3. Partial derivative of x_proj with respect to x_proj_hom (projected in homogenous coordinates)
        dxproj_dxprojhom = dxproj_dxprojhom_matrix
        # Combining partial derivaties for chain rule
        dxproj_dti = dxproj_dxprojhom @ dxprojhom_dpc @ dpc_dti
        # Put into Jacobian
        J[:,i] = dxproj_dti.flatten()
    
    # Rotation Parameters Jacobian Matrix
    # Retrieve the rotation parameters from the rotation matrix
    rotation_params = rpy_from_dcm(C_wc)
    roll = rotation_params[0,0]
    pitch = rotation_params[1,0]
    yaw = rotation_params[2,0]

    # Compute Partial Derivatives of Rotation Matrix
    # Note: Same 3 partial derivatives (2 and 3 remain same; 1 now respect to rotational components not translational)
    # 1. Camera Coordinates: pc = Ccw(pw - t)
    # 2. Projection: x_proj_hom = K * pc
    # 3. Normalization: x_proj = x_proj_hom / x_proj_hom[2]

    # Compute Camera Coordinates Partial Derivative
    # Using Product Rule we find that the derivative is dCcw/dri * (Wpt - t) where ri is the rotational component
    # 1a) Computing Derivatives of rotation matrix with respect to roll, pitch, yaw

    # Compute ZYX Euler Angle Matrices for roll(x), pitch(y) and yaw(z)
    # Roll (x-axis rotation)
    Croll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    # Pitch (y-axis rotation)
    Cpitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    # Yaw (z-axis rotation)
    Cyaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])

    # Compute the derivatives of each of these matrices with respect to each other
    # Derivative of roll rotation with respect to roll
    dCroll_droll = np.array([
        [0, 0, 0],
        [0, -np.sin(roll), -np.cos(roll)],
        [0, np.cos(roll), -np.sin(roll)]
    ])
    # Derivative of pitch rotation with respect to pitch
    dCpitch_dpitch = np.array([
        [-np.sin(pitch), 0, np.cos(pitch)],
        [0, 0, 0],
        [-np.cos(pitch), 0, -np.sin(pitch)]
    ])
    # Derivative of yaw rotation with respect to yaw
    dCyaw_dyaw = np.array([
        [-np.sin(yaw), -np.cos(yaw), 0],
        [np.cos(yaw),  -np.sin(yaw), 0],
        [0, 0, 0]
    ])


    dCcw_droll = (Cyaw @ Cpitch @ dCroll_droll).T
    dCcw_dpitch = (Cyaw @ dCpitch_dpitch @ Croll).T
    dCcw_dyaw = (dCyaw_dyaw @ Cpitch @ Croll).T

    # 1b) Compute Full Camera Coordinate Partial Derivative
    delta = Wpt - t
    dpc_droll = dCcw_droll @ delta
    dpc_dpitch = dCcw_dpitch @ delta
    dpc_dyaw = dCcw_dyaw @ delta

    # Combining 3 partial derivatives
    dxproj_droll = dxproj_dxprojhom @ dxprojhom_dpc @ dpc_droll
    dxproj_dpitch = dxproj_dxprojhom @ dxprojhom_dpc @ dpc_dpitch
    dxproj_dyaw = dxproj_dxprojhom @ dxprojhom_dpc @ dpc_dyaw

    # Poplulate Jacobian
    J[:,3] = dxproj_droll.flatten()
    J[:,4] = dxproj_dpitch.flatten()
    J[:,5] = dxproj_dyaw.flatten()




    correct = isinstance(J, np.ndarray) and \
        J.dtype == np.float64 and J.shape == (2, 6)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return J


def rpy_from_dcm(R):
    """
    Roll, pitch, yaw Euler angles from rotation matrix.

    The function computes roll, pitch and yaw angles from the
    rotation matrix R. The pitch angle p is constrained to the range
    (-pi/2, pi/2].  The returned angles are in radians.

    Inputs:
    -------
    R  - 3x3 orthonormal rotation matrix.

    Returns:
    --------
    rpy  - 3x1 np.array of roll, pitch, yaw Euler angles.
    """
    rpy = np.zeros((3, 1))

    # Roll.
    rpy[0] = np.arctan2(R[2, 1], R[2, 2])

    # Pitch.
    sp = -R[2, 0]
    cp = np.sqrt(R[0, 0]*R[0, 0] + R[1, 0]*R[1, 0])

    if np.abs(cp) > 1e-15:
      rpy[1] = np.arctan2(sp, cp)
    else:
      # Gimbal lock...
      rpy[1] = np.pi/2
  
      if sp < 0:
        rpy[1] = -rpy[1]

    # Yaw.
    rpy[2] = np.arctan2(R[1, 0], R[0, 0])

    return rpy