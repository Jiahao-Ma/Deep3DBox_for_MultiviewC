#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
import random

from numpy.lib.function_base import angle

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))
    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def eulerAnglesToRotationMatrix(theta) :
    
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
        
        
                    
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                    
                    
    R = np.dot(R_z, np.dot( R_y, R_x ))

    

    return R


if __name__ == '__main__' :
    import json

    e = np.random.rand(3) * math.pi * 2 - math.pi
    angle_xyz = [0.185417, -21.171631, 133.861435]
    print('angle_xyz:', angle_xyz)
    angle_xyz = [ np.deg2rad(x) for x in angle_xyz]
    print('angle_xyz:', angle_xyz)
    e = angle_xyz
    
    R = eulerAnglesToRotationMatrix(e)
    e1 = rotationMatrixToEulerAngles(R)

    R1 = eulerAnglesToRotationMatrix(e1)
    print ("\nInput Euler angles :\n{0}".format(e))
    print ("\nR :\n{0}".format(R))
    print ("\nOutput Euler angles :\n{0}".format(e1))
    print ("\nR1 :\n{0}".format(R1))

    p = r'F:\ANU\ENGN8602\Data\MultiviewC_dataset\calibrations\Camera1\parameters.json'
    def get_R(calib_path):
        with open(calib_path, 'r') as f:
            params = json.loads(f.read())
        return params['R']
    R = np.array(get_R(p))
    print('R:\n',R.reshape(3,3))
    [x,y,z] = rotationMatrixToEulerAngles(R)
    print('x:',np.rad2deg(x))
    print('y:',np.rad2deg(y))
    print('z:',np.rad2deg(z))
