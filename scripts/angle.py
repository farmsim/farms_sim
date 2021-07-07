# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:39:20 2021

@author: guila
"""
import numpy as np

def quaternion_to_euler(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)

    t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
    Y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.arctan2(t3, t4)

    return X, Y, Z

def euler_to_quaternion(phi, theta, psi):

    q0 = np.cos(phi/2)*np.cos(theta/2)*np.cos(psi/2)+\
    np.sin(phi/2)*np.sin(theta/2)*np.sin(psi/2)

    q1 = np.sin(phi/2)*np.cos(theta/2)*np.cos(psi/2)-\
    np.cos(phi/2)*np.sin(theta/2)*np.sin(psi/2)

    q2 = np.cos(phi/2)*np.sin(theta/2)*np.cos(psi/2)+\
    np.sin(phi/2)*np.cos(theta/2)*np.sin(psi/2)

    q3 = np.cos(phi/2)*np.cos(theta/2)*np.sin(psi/2)+\
    np.sin(phi/2)*np.sin(theta/2)*np.cos(psi/2)

    return q0, q1, q2, q3
