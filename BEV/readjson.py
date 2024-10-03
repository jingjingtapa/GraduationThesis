import numpy as np
 
def quaternion_rotation_matrix(Q):
    q0, q1, q2, q3 = Q
    R = np.array([[2 * (q0 * q0 + q1 * q1) - 1,  2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
                            [2 * (q1 * q2 + q0 * q3), 2 * (q0 * q0 + q2 * q2) - 1, 2 * (q2 * q3 - q0 * q1)],
                            [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 2 * (q0 * q0 + q3 * q3) - 1]])
                            
    return R
def quarternion_to_rotation(q):
    qx, qy, qz, qw = q
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

Q = [ 0.9735839424380041,
    -0.010751769161021867,
    0.0027191710555974913,
    0.22805988817753894]

print(quarternion_to_rotation(Q))
print(quaternion_rotation_matrix(Q))