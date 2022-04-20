import math as m
import numpy as np
import tf.transformations as ts
import cv2
import rospy

def project_points(pts, camera_position):
    xc = camera_position[0][0]
    yc = camera_position[1][0]
    zc = camera_position[2][0]

    denom = zc - pts[2]

    transform1 = np.array([[zc, 0, -xc],
                        [0, zc, -yc]])

    pts = np.matmul(transform1, pts)
    return np.divide(pts, [denom, denom])

def cartesian_cross_product(x,y):
    cross_product = np.transpose([np.tile(x, len(y)),np.repeat(y,len(x))])
    return cross_product

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return np.array(rot_matrix)

def geometry_msgs_TransformStamped_to_pos_rot(transform):
    rotation = transform.transform.rotation
    translation = transform.transform.translation

    rot_quat = np.array([rotation.w, rotation.x, rotation.y, rotation.z])
    pos_transform = pos = np.array([[translation.x], [translation.y], [translation.z]])

    rot_matrix = quaternion_rotation_matrix(rot_quat)

    return pos_transform, rot_matrix

def htm_to_pos_rot(htm):
    pos = np.array([ts.translation_from_matrix(htm)]).T
    rot = htm[0:3,0:3]
    return pos, rot

def pos_rot_to_htm(pos, rot):
    return np.vstack((np.hstack((rot, pos)), np.array([0, 0, 0, 1])))

def geometry_msgs_TransformStamped_to_htm(transform_stamped):
    trans = transform_stamped.transform.translation
    rot = transform_stamped.transform.rotation
    trans = [trans.x, trans.y, trans.z]
    rot = [rot.x, rot.y, rot.z, rot.w]
    return ts.concatenate_matrices(ts.translation_matrix(trans), ts.quaternion_matrix(rot))

def vertical_to_horizontal_fov(width, height, vertical_fov):
    f = (height/2)*m.tan(vertical_fov)
    return 2*m.atan(width/2/f)

def imshow_r(window_name, img, window_size):
    img = np.copy(img)
    img_height, img_width, img_depth = np.shape(img)
    window_width, window_height = window_size
    window_aspect = window_width/window_height
    img_aspect = img_width/img_height
    if img_aspect>window_aspect:
        out_width = window_width
        out_height = int(out_width/img_aspect)
    else:
        out_height = window_height
        out_width = int(out_height*img_aspect)
    resized = cv2.resize(img, (out_width, out_height))

    
    ret = np.zeros((window_height, window_width, img_depth), dtype=img.dtype)
    ret[-out_height//2+window_height//2:out_height//2+window_height//2, -out_width//2+window_width//2:out_width//2+window_width//2, :] = resized[:,:,:]
    cv2.imshow(window_name, ret)
    return ret

def duration_to_sec(duration):
    return duration.secs + duration.nsecs/10**9

class Times():
    def __init__(self):
        self.times = []
        self.add("begin")

    def add(self):
        self.times.append((str(len(self.times)), rospy.Time.now()))
    
    def add(self, name):
        self.times.append((str(name), rospy.Time.now()))
    
    def __str__(self):
        ret = ""
        dt = duration_to_sec(self.times[-1][1]-self.times[0][1])
        for i in range(1,len(self.times)):
            ret += "{:s}: {:.2f}%\n".format(self.times[i][0], duration_to_sec(self.times[i][1]-self.times[i-1][1])/dt*100)
        ret += "Total: {:.2f}, {:.2f} hz\n".format(dt, 1/dt)
        return ret

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         m.cos(theta[0]), -m.sin(theta[0]) ],
                    [0,         m.sin(theta[0]), m.cos(theta[0])  ]
                    ])

    R_y = np.array([[m.cos(theta[1]),    0,      m.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-m.sin(theta[1]),   0,      m.cos(theta[1])  ]
                    ])

    R_z = np.array([[m.cos(theta[2]),    -m.sin(theta[2]),    0],
                    [m.sin(theta[2]),    m.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-4

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = m.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = m.atan2(R[2,1] , R[2,2])
        y = m.atan2(-R[2,0], sy)
        z = m.atan2(R[1,0], R[0,0])
    else :
        x = m.atan2(-R[1,2], R[1,1])
        y = m.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])


def extrapolate_htm(stamp1, htm1, stamp2, htm2, target_stamp):
    pos1, rot1 = htm_to_pos_rot(htm1)
    pos2, rot2 = htm_to_pos_rot(htm2)

    stamp1_to_target = duration_to_sec(target_stamp-stamp1)
    stamp1_to_stamp2 = duration_to_sec(stamp2-stamp1)

    velocity = (pos2-pos1)/stamp1_to_stamp2
    rotation = np.matmul(rot2, np.linalg.inv(rot1))

    angular_velocity = rotationMatrixToEulerAngles(rotation)*(stamp1_to_stamp2)

    new_position = pos1 + velocity*stamp1_to_target
    new_rotation = np.matmul(eulerAnglesToRotationMatrix(angular_velocity*stamp1_to_stamp2), rot1)

    return pos_rot_to_htm(new_position, new_rotation)




