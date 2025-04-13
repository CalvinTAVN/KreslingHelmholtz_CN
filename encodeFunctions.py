import spidev
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import can
from scipy.spatial.transform import Rotation as R
import sys

#given a set of values to encodeNum for the 6 coils
#convert them to binary
def encodeNum(values):
	sign = 0
	for i in range(6):
		sign = sign + (values[i]<0)*2**i
	arr = [sign, abs(2*values[0]), abs(2*values[1]), abs(2*values[2]), abs(2*values[3]), abs(2*values[4]), abs(2*values[5]),0]
	output  = [int(i) for i in arr]
	return output

#send zeros
def zero():	
	values = [0,0,0,0,0,0]
	tx = encodeNum(values)
	message = can.Message(arbitration_id=0x00, is_extended_id=False, data= tx)
	bus.send(message, timeout=0.5)
	time.sleep(0.01)
	return	

def sinwave(t, A, f, phi):
	t = np.array(t)
	out = A*np.sin(2*np.pi*f*t + phi)
	return out
	
def coswave(t, A, f, phi):
	t = np.array(t)
	out = A*np.cos(2*np.pi*f*t + phi)
	return out
	
def round2half(arr):
	return [round(2*i)/2 for i in arr] 

def constant(t, A):	
	return A*np.ones(len(t))

def roll_yz(step_no, speed, init, A, Ts, direction=0):
    """
    Roll magnetization vector around the X-axis (i.e., in the YZ plane).
    
    Parameters:
        step_no (int): Number of quarter turns (e.g., 4 = full roll)
        speed (float): Frequency of rotation in Hz (rotations per second)
        init (list): Initial magnetization vector (3D)
        A (float): Amplitude of magnetization
        Ts (float): Sampling time (s)
        direction (str): 'CW' (clockwise) or 'CCW' (counterclockwise) viewed along +X axis
        
    Returns:
        [x, y, z, final_state]: Arrays of magnetization over time, and final magnetization vector
    """
    k = int(1 / (4 * speed * Ts))  # samples per quarter roll
    n = k * step_no
    t = np.arange(n) * Ts

    init_vec = np.array(init)

    # Rotation axis: X-axis
    roll_axis = np.array([1, 0, 0])
    if direction == 1: #CW
        roll_axis = -roll_axis  # flip axis to reverse rotation direction

    # Rotation angles over time
    rotation_angles = 2 * np.pi * speed * t

    # Create rotation vectors: angle * axis
    rotation_vectors = np.outer(rotation_angles, roll_axis)

    # Apply rotation to initial vector
    rotations = R.from_rotvec(rotation_vectors)
    magnetization = rotations.apply(init_vec)

    x, y, z = A * magnetization.T

    # Final state after all rotations
    final_rotation = R.from_rotvec(rotation_vectors[-1])
    state_vec = final_rotation.apply(init_vec)
    state = state_vec.round(decimals=2).tolist()

    return [x, y, z, state]

#rotate in the X-Y plane
def rotate(angle, direction, speed, init, A, Ts):
	#angle of magnetization 
	theta0 = np.arctan2(init[1], init[0])
	print("Initial Angle is : ", theta0*180/np.pi)
	if (theta0 < 0):
		theta0 = theta0 + 2*np.pi

	dphi = angle - theta0

	if (direction == 1):  #if clockwise turn
		if (dphi < 0):
			dphi = dphi + 2*np.pi
		n = int(dphi/(2*np.pi*speed*Ts))
		t = np.arange(n)*Ts
		
		print('dphi: ', dphi)
		print('n: ', n)
		
		z = np.zeros(n)
		x = A*np.cos(2*np.pi*speed*t + theta0)
		y = A*np.sin(2*np.pi*speed*t + theta0)
	elif (direction == -1):  #if counter-clockwise turn	
		if (dphi > 0):
			dphi = 2*np.pi - dphi
		else:
			dphi = abs(dphi)
		n = int(dphi/(2*np.pi*speed*Ts))
		t = np.arange(n)*Ts
		
		z = np.zeros(n)
		x = A*np.cos(-2*np.pi*speed*t + theta0)
		y = A*np.sin(-2*np.pi*speed*t + theta0)
		
	state = [np.cos(angle), np.sin(angle), 0]
	
	return [x,y,-z,state]

def con(vector, n ):
	x = vector[0]*np.ones(n)
	y = vector[1]*np.ones(n)
	z = vector[2]*np.ones(n)
	
	return [x, x, y, y, z, z]

def sendCAN(x, y, z, can, bus):
	print("Sequence Started")
	x = round2half(x)
	y = round2half(y)
	z = round2half(z)
	for i in range(len(x)):
		values = [x[i], x[i], y[i], y[i], z[i], z[i]]
		tx = encodeNum(values)
		message = can.Message(arbitration_id=0x00, is_extended_id=False, data= tx)
		bus.send(message, timeout=0.5)
		time.sleep(0.01)
	values = [0,0,0,0,0,0]
	tx = encodeNum(values)
	message = can.Message(arbitration_id=0x00, is_extended_id=False, data= tx)
	bus.send(message, timeout=0.5)
	print("Sent back to 0")
	time.sleep(0.01)