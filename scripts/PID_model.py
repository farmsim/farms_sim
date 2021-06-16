# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:30:13 2021

@author: Caner Durmusoglu

modification: Guilain Brunoro
"""

import time
import numpy as np

class PID:
    """PID Controller
    """

    def __init__(self, P=0.2, I=0.0, D=0.0, current_time=None):

        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.sample_time = 0.00
        self.current_time = current_time if current_time is not None else time.time()
        self.last_time = self.current_time

        self.clear()

    def clear(self):
        """Clears PID computations and coefficients"""
        self.SetPoint = 0.0

        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0

        # Windup Guard
        self.int_error = 0.0
        self.windup_guard = 0.2

        self.output = 0.0

    def update(self, feedback_value, current_time=None):
        """Calculates PID value for given reference feedback
        .. math::
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}
           Test PID with Kp=1.2, Ki=1, Kd=0.001 (test_pid.py)
        """
        feedback_value = clip_map(feedback_value) #constraint the angle between [-pi,pi]
        
        if abs(self.SetPoint - feedback_value) > np.pi:
            
            
            
            # self.SetPoint = (self.SetPoint + 2*np.pi) % (2*np.pi)
            # feedback_value = (feedback_value + 2*np.pi) % (2*np.pi)
                
            if self.SetPoint <0:
                self.SetPoint = self.SetPoint+np.pi
            else:
                self.SetPoint = self.SetPoint-np.pi
                
            if feedback_value <0:
                feedback_value = feedback_value+np.pi
            else:
                feedback_value = feedback_value-np.pi

            # if self.SetPoint<0:
            #     self.SetPoint = -np.arccos(-np.cos(self.SetPoint))
            # else:
            #     self.SetPoint = np.arcsin(-np.sin(np.arccos(-np.cos(self.SetPoint))))
            # if feedback_value<0:
            #     feedback_value = -np.arccos(-np.cos(feedback_value))
            # else:
            #     feedback_value = np.arcsin(-np.sin(np.arccos(-np.cos(feedback_value))))
             
        error = self.SetPoint - feedback_value

        self.current_time = current_time if current_time is not None else time.time()
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error

        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error
            self.ITerm += error * delta_time

            # anti-windup regulation
            self.Iterm = np.clip(self.ITerm,-self.windup_guard,self.windup_guard)

            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time

            # Remember last time and last error for next calculation
            self.last_time = self.current_time
            self.last_error = error

            self.output = -(self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm))
            self.output = np.clip(self.output,-0.2,0.2)

    def setKp(self, proportional_gain):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        """Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        """Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
        self.Kd = derivative_gain

    def setWindup(self, windup):
        """Integral windup, also known as integrator windup or reset windup,
        refers to the situation in a PID feedback controller where
        a large change in setpoint occurs (say a positive change)
        and the integral terms accumulates a significant error
        during the rise (windup), thus overshooting and continuing
        to increase as this accumulated error is unwound
        (offset by errors in the other direction).
        The specific problem is the excess overshooting.
        """
        self.windup_guard = windup

    def setSampleTime(self, sample_time):
        """PID that should be updated at a regular interval.
        Based on a pre-determined sampe time, the PID decides if it should compute or return immediately.
        """
        self.sample_time = sample_time
        

def clip_map(angle):
    if angle > 0:
        if angle > np.pi:
            return angle - 2*np.pi
    else:
        if angle < -np.pi:
            return angle + 2*np.pi
    return angle