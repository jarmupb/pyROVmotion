# -*- coding: utf-8 -*-
"""
Created on Mon Feb 06 15:03:06 2017

@author: UPB
"""

#==============================================================================
# Kinematics module
#==============================================================================
# Author:   Juan A. Ramírez-Macías
# email:    juan.ramirez@upb.edu.co
# version:  v0.0
# date:     2017-02-06
#------------------------------------------------------------------------------
# Universidad Pontificia Bolivariana
# Grupo de Automática y Diseño A+D
#==============================================================================
# Purpose:
#------------------------------------------------------------------------------
# This module is a function suite with kinematics tools for modelling
# underwater systems' motion
#
#------------------------------------------------------------------------------
# Required modules and functions:
#------------------------------------------------------------------------------
# This module requires the following modules: NumPy
#
#------------------------------------------------------------------------------
# Included Functions:
#------------------------------------------------------------------------------
# coriolis: calculates a Coriolis matrix from a 6x6 mass matrix and a velocity
#           vector
# dir2mat: obtains one 3x3 rotation matrix from a 1x3 unitary vector
# euler2mat: obtains a 3x3 rotation matrix from Euler angles phi, theta, psi
# mat2euler: obtains Euler angles phi, theta, psi from a 3x3 rotation matrix
# mat2quat: obtains a 1x4 quaternion from a 3x3 rotation matrix
# quat2euler: obtains Euler angles phi, theta, psi from a 1x4 quaternion
# quat2mat: obtains a 3x3 rotation matrix from a 1x4 quaternion
# skew: obtains a 3x3 skew-symmetric matrix from a 1x3 vector
#==============================================================================

#==============================================================================
# Required modules' functions
#==============================================================================
from numpy import amax, arcsin, arctan2, argmax, array, cos, dot, eye, sin, \
                  sqrt, zeros

#==============================================================================
# Coriolis matrix calculation
#==============================================================================
def coriolis(M, v):
    # Coriolis preallocation
    C = zeros([6, 6])
    # Matrix assembly
    C[:3,3:] = -skew(dot(v[:3], M[:3,:3].T) + dot(v[3:], M[:3,3:].T))
    C[3:,:3] = C[:3,3:]
    C[3:,3:] = -skew(dot(v[:3], M[3:,:3].T) + dot(v[3:], M[3:,3:].T))
    return C

#==============================================================================
# Rotation matrix from unitary vector
#==============================================================================
def dir2mat(u):
    # Using euler angles to define the orientation
    # theta
    theta_i = arctan2(u[0], u[2])
    cq = cos(theta_i)
    sq = sin(theta_i)
    # phi
    if cq >= sq:
        phi_i = arctan2(-u[1], u[2] / cq)
    else:
        phi_i = arctan2(-u[1], u[0] / sq)
    cp = cos(phi_i)
    sp = sin(phi_i)
    # Rotation matrix
    return array([[cq, sq * sp, sq * cp],
                  [0, cp, -sp],
                  [-sq, cq * sp, cq * cp]])

#==============================================================================
# Rotation matrix from Euler angles
#==============================================================================
def euler2mat(phi, theta, psi):
    
    # Sines and cosines
    c_phi = cos(phi)
    s_phi = sin(phi)
    c_theta = cos(theta)
    s_theta = sin(theta)
    c_psi = cos(psi)
    s_psi = sin(psi)
    
    # Return rotation matrix
    return array([
        [c_psi * c_theta, -s_psi * c_phi + c_psi * s_theta * s_phi,
                                     s_psi * s_phi + c_psi * c_phi * s_theta],
        [s_psi * c_theta, c_psi * c_phi + s_phi * s_theta * s_psi,
                                    -c_psi * s_phi + s_theta * s_psi * c_phi],
        [-s_theta, c_theta * s_phi, c_theta * c_phi]])
        
#==============================================================================
# Euler angles from rotation matrix
#==============================================================================
def mat2euler(Ri):
    return array([arctan2(Ri[2,1], Ri[2,2]),    # phi
                  -arcsin(Ri[2,0]),             # theta
                  arctan2(Ri[1,0], Ri[0,0])])   # psi

#==============================================================================
# Quaternions from rotation matrix
#==============================================================================
def mat2quat(R):
    
    R44 = R[0,0] + R[1,1] + R[2,2]
    trR = array([R[0,0], R[1,1], R[2,2], R44])

    Rii = amax(trR)
    ii = argmax(trR)
    
    p_i= sqrt(1 + 2 * Rii - R44);
    if ii == 0:
       p1 = p_i
       p2 = (R[1,0] + R[0,1]) / p_i
       p3 = (R[0,2] + R[2,0]) / p_i
       p4 = (R[2,1] - R[1,2]) / p_i
    elif ii == 1:
       p1 = (R[1,0] + R[0,1]) / p_i
       p2 = p_i
       p3 = (R[2,1] + R[1,2]) / p_i
       p4 = (R[0,2] - R[2,0]) / p_i
    elif ii == 2:
       p1 = (R[0,2] + R[2,0]) / p_i
       p2 = (R[2,1] + R[1,2]) / p_i   
       p3 = p_i
       p4 = (R[1,0] - R[0,1]) / p_i   
    else:
       p1 = (R[2,1] - R[1,2]) / p_i
       p2 = (R[0,2] - R[2,0]) / p_i
       p3 = (R[1,0] - R[0,1]) / p_i   
       p4 = p_i
    
    e = 0.5 * array([p4, p1, p2, p3])
    return e / dot(e, e)

#==============================================================================
# Euler angles from quaternions
#==============================================================================
def quat2euler(e):
    Ri = quat2mat(e)
    return mat2euler(Ri)
    
#==============================================================================
# Rotation matrix from quaternions
#==============================================================================
def quat2mat(e):
    qq1 = e[1] ** 2
    qq2 = e[2] ** 2
    qq3 = e[3] ** 2
    q0q1 = e[0] * e[1]
    q0q2 = e[0] * e[2]
    q0q3 = e[0] * e[3]
    q1q2 = e[1] * e[2]
    q1q3 = e[1] * e[3]
    q2q3 = e[2] * e[3]
    
    return array([[1-2*(qq2 + qq3), 2*(q1q2 - q0q3), 2*(q1q3 + q0q2)], 
                  [2*(q1q2 + q0q3), 1-2*(qq1 + qq3), 2*(q2q3 - q0q1)],
                  [2*(q1q3 - q0q2), 2*(q2q3 + q0q1), 1-2*(qq1 + qq2)]])

#==============================================================================
# Skew-symmetric matrix from vector
#==============================================================================
def skew(r):
    return array([[0., -r[2], r[1]],
                  [r[2], 0., -r[0]],
                  [-r[1], r[0], 0.]])
                  
#==============================================================================
# Kinematic transformation when quaternions are used and velocity is measured
# in the global frame
#==============================================================================
def T_q(q):
    T = zeros([7, 6])
    T[:3,:3] = eye(3)
    T[3:,3:] = 0.5 * array([[-q[4],-q[5],-q[6]],
                            [ q[3], q[6],-q[5]],
                            [-q[6], q[3], q[4]],
                            [ q[5],-q[4], q[3]]])
    return T
    
#==============================================================================
# Kinematic transformation when Euler angles are used and velocity is measured
# in the body frame
#==============================================================================
def J_eta(eta):
    # Attitude
    phi = eta[3]
    theta = eta[4]
    psi = eta[5]
    
    # Sines and cosines
    c_phi = cos(phi)
    s_phi = sin(phi)
    c_theta = cos(theta)
    s_theta = sin(theta)
    c_psi = cos(psi)
    s_psi = sin(psi)
    
    # Matrix preallocation
    J = zeros([6, 6])
    
    # Matrix assembly
    # Rotation matrix
    J[:3,:3] = array([
        [c_psi * c_theta, -s_psi * c_phi + c_psi * s_theta * s_phi,
                                     s_psi * s_phi + c_psi * c_phi * s_theta],
        [s_psi * c_theta,  c_psi * c_phi + s_phi * s_theta * s_psi,
                                    -c_psi * s_phi + s_theta * s_psi * c_phi],
        [-s_theta, c_theta * s_phi, c_theta * c_phi]])
    # Angular velocities transformation
    J[3:,3:] = array([
                    [1., s_phi * s_theta / c_theta, c_phi * s_theta / c_theta],
                    [0., c_phi, -s_phi],
                    [0., s_phi / c_theta, c_phi / c_theta]])
    return J

#==============================================================================
# Kinematic transformation when Euler angles are used and velocity is measured
# in the body frame
#==============================================================================
def J_psi(psi):
    # Attitude
    phi = 0.
    theta = 0.
    
    # Sines and cosines
    c_phi = cos(phi)
    s_phi = sin(phi)
    c_theta = cos(theta)
    s_theta = sin(theta)
    c_psi = cos(psi)
    s_psi = sin(psi)
    
    # Matrix preallocation
    J = zeros([6, 6])
    
    # Matrix assembly
    # Rotation matrix
    J[:3,:3] = array([
        [c_psi * c_theta, -s_psi * c_phi + c_psi * s_theta * s_phi,
                                     s_psi * s_phi + c_psi * c_phi * s_theta],
        [s_psi * c_theta,  c_psi * c_phi + s_phi * s_theta * s_psi,
                                    -c_psi * s_phi + s_theta * s_psi * c_phi],
        [-s_theta, c_theta * s_phi, c_theta * c_phi]])
    # Angular velocities transformation
    J[3:,3:] = array([
                    [1., s_phi * s_theta / c_theta, c_phi * s_theta / c_theta],
                    [0., c_phi, -s_phi],
                    [0., s_phi / c_theta, c_phi / c_theta]])
    return J

#==============================================================================
# Basic rotation matrices
#==============================================================================
def R_x(ang):
    return array([[1., 0., 0.], 
                  [0., cos(ang), -sin(ang)],
                  [0., sin(ang),  cos(ang)]])
def R_y(ang):
    return array([[ cos(ang), 0., sin(ang)],
                  [0., 1., 0.],
                  [-sin(ang), 0., cos(ang)]])
def R_z(ang):
    return array([[cos(ang), -sin(ang), 0.],
                  [sin(ang),  cos(ang), 0.],
                  [0., 0., 1.]])