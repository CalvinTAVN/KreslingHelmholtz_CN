import spidev
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import can
import sys




def encode(values):
	sign = 0
	for i in range(6):
		sign = sign + (values[i]<0)*2**i
	arr = [sign, abs(2*values[0]), abs(2*values[1]), abs(2*values[2]), abs(2*values[3]), abs(2*values[4]), abs(2*values[5]),0]
	output  = [int(i) for i in arr]
	return output
	
def div(direction, sign, n, A ):

	out = sign*A*np.ones(n)
	zer = np.zeros(n)
	
	if (direction == 'x'):
		return [out, -out, zer, zer, zer, zer]
	elif (direction == 'y'):
		return [zer, zer, out, -out, zer, zer]
	elif (direction == 'z'):
		return [zer, zer, zer, zer, out, -out]
	else:
		print('Not valid direction given')
		return


def con(vector, n ):

	x = vector[0]*np.ones(n)
	y = vector[1]*np.ones(n)
	z = vector[2]*np.ones(n)
	
	return [x, x, y, y, z, z]

def rotate(angle, direction, speed, init, A, Ts):
	theta0 = np.arctan2(init[1], init[0])
	print("Initial Angle is: ", theta0*180/np.pi)
	
	if (theta0 < 0):
		theta0 = theta0 + 2*np.pi
	
	angle = angle*np.pi/180.0
	if (angle < 0):
		angle = angle + 2*np.pi
	
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
	


def roll(step_no, angle, speed, init, A, Ts):
	k = int(1/(4*speed*Ts))  #number of samples per quarter roll
	[x0, y0, z0] = init #initial magnetization orientation
	
	theta = angle*np.pi / 180
	Ax = A*np.cos(theta)
	Ay = A*np.sin(theta)
	
	n = k*step_no
	t = np.arange(n)*Ts
	
	tolerance = 0.01
	
	x_state = 0
	y_state = 0
	z_state = 0

	if (z0 == 1 or z0 == -1):
		if (step_no%4 == 0):
			z_state = z0
		elif (step_no%4 == 2):
			z_state = - z0
		elif (step_no%4 == 1):
			z_state = 0
			x_state = z0*np.cos(theta)
			y_state = z0*np.sin(theta)
		elif (step_no%4 == 3):
			z_state = 0
			x_state = -z0*np.cos(theta)
			y_state = -z0*np.sin(theta)			

	
	if (z0 == 1):  #if magnetization points up, the x0,y0 are irrelevant.
		z = A*np.cos(2*np.pi*speed*t)
		x = Ax*np.sin(2*np.pi*speed*t)
		y = Ay*np.sin(2*np.pi*speed*t)
	elif (z0 == -1):  #if magnetization points down, the x0,y0 are irrelevant.
		z = -A*np.cos(2*np.pi*speed*t)
		x = -Ax*np.sin(2*np.pi*speed*t)
		y = -Ay*np.sin(2*np.pi*speed*t)
		
	if (z0 == 0):
		if ( abs(np.cos(theta)-x0) < tolerance and abs(np.sin(theta)-y0) < tolerance ):
			
			z = -A*np.sin(2*np.pi*speed*t)
			x = Ax*np.cos(2*np.pi*speed*t)
			y = Ay*np.cos(2*np.pi*speed*t)
				
			if (step_no%4 == 0):
				x_state = x0
				y_state = y0
				z_state = z0
				
			elif (step_no%4 == 2):
				x_state = -x0
				y_state = -y0
				z_state = z0
			elif (step_no%4 == 3):
				z_state = 1
				x_state = 0
				y_state = 0
			elif (step_no%4 == 1):
				z_state = -1
				x_state = 0
				y_state = 0
				
		elif ( abs(np.cos(theta)+x0) < tolerance and abs(np.sin(theta)+y0) < tolerance ):
			
			z = A*np.sin(2*np.pi*speed*t)
			x = -Ax*np.cos(2*np.pi*speed*t)
			y = -Ay*np.cos(2*np.pi*speed*t)
				
			if (step_no%4 == 0):
				x_state = x0
				y_state = y0
				z_state = z0
				
			elif (step_no%4 == 2):
				x_state = -x0
				y_state = -y0
				z_state = z0
			elif (step_no%4 == 3):
				z_state = -1
				x_state = 0
				y_state = 0
			elif (step_no%4 == 1):
				z_state = 1
				x_state = 0
				y_state = 0		
		else:
			print("Initial orientation is different than step agle. Do a rotation first")
			return [[],[],[],init]
	
	state = [x_state, y_state, z_state]
	return [x,y,-z,state]
	
	
def zero():	
	values = [0,0,0,0,0,0]
	tx = encode(values)
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



Ts = 0.01

print("hello world")


bus = can.interface.Bus(bustype='socketcan', channel='can0', bitrate=1000000)
print("Canbus Successfully Setup. \n")


values = [0,0,0,0,0,0]
tx = encode(values)
message = can.Message(arbitration_id=0x00, is_extended_id=False, data= tx)
bus.send(message, timeout=0.5)
time.sleep(0.01)


print("Please provide initial state:")
x0 = input("Enter x0:")
y0 = input("Enter y0:")
z0 = input("Enter z0:")
state = [float(x0), float(y0), float(z0)]
print("Initial state:", state)



while True:
	motion = input("Enter 'r' for rolling, 't' for rotation, 'd' for divergent field, 'c' for constant field,  or 's' to stop:")
	if (motion == 'r'):
	
		step_no = input("Number of steps:")
		step_no = int(step_no)
		
		angle = input("Angle:")
		angle = int(angle)
		
		speed = input("Speed:")
		speed = float(speed)
		
		A = input("Amplitude:")
		A = int(A)
		
		[x, y, z,state] = roll(step_no, angle, speed, state, A, Ts)

		x = round2half(x)
		y = round2half(y)
		z = round2half(z)
		
		print("Sequence Started")
		for i in range(len(x)):
			values = [x[i], x[i], y[i], y[i], z[i], z[i]]
			tx = encode(values)
			message = can.Message(arbitration_id=0x00, is_extended_id=False, data= tx)
			bus.send(message, timeout=0.5)
			time.sleep(0.01)
		print('State:', state)
	elif (motion == 's'):
		break
	elif (motion == 'd'):
	
		
		direction = input("Direction:")
		direction = str(direction)
		
		sign = input("Sign:")
		sign = int(sign)
		
		n = input("Number of samples:")
		n = int(n)
		
		A = input("Amplitude:")
		A = int(A)
		
		[x1, x2, y1, y2, z1, z2] = div(direction, sign, n, A )

		x1 = round2half(x1)
		x2 = round2half(x2)
		
		y1 = round2half(y1)
		y2 = round2half(y2)
		
		z1 = round2half(z1)
		z2 = round2half(z2)


		print("Sequence Started")
		for i in range(len(x1)):
			values = [x1[i], x2[i], y1[i], y2[i], z1[i], z2[i]]
			tx = encode(values)
			message = can.Message(arbitration_id=0x00, is_extended_id=False, data= tx)
			bus.send(message, timeout=0.5)
			time.sleep(0.01)
		print('State:', state)
	elif (motion == 'c'):
		x = input("x:")
		x = int(x)
		
		y = input("y:")
		y = int(y)
		
		z = input("z:")
		z = int(z)
		
		
		n = input("Number of samples:")
		n = int(n)
		
		
		[x1, x2, y1, y2, z1, z2] = con([x,y,z], n)

		x1 = round2half(x1)
		x2 = round2half(x2)
		
		y1 = round2half(y1)
		y2 = round2half(y2)
		
		z1 = round2half(z1)
		z2 = round2half(z2)


		print("Sequence Started")
		for i in range(len(x1)):
			values = [x1[i], x2[i], y1[i], y2[i], z1[i], z2[i]]
			tx = encode(values)
			message = can.Message(arbitration_id=0x00, is_extended_id=False, data= tx)
			bus.send(message, timeout=0.5)
			time.sleep(0.01)
		print('State:', state)
	elif (motion == 't'):
		angle = input("angle:")
		angle = int(angle)
		
		direction = input("direction:")
		direction = int(direction)
		
		speed = input("Speed:")
		speed = float(speed)
		
		A = input("Amplitude:")
		A = int(A)
		

		[x, y, z,state] = rotate(angle, direction, speed, state, A, Ts)

		x = round2half(x)
		y = round2half(y)
		z = round2half(z)
		
		print("Sequence Started")
		for i in range(len(x)):
			values = [x[i], x[i], y[i], y[i], z[i], z[i]]
			tx = encode(values)
			message = can.Message(arbitration_id=0x00, is_extended_id=False, data= tx)
			bus.send(message, timeout=0.5)
			time.sleep(0.01)
		print('State:', state)


	zero()
		




values = [0,0,0,0,0,0]
tx = encode(values)
message = can.Message(arbitration_id=0x00, is_extended_id=False, data= tx)
bus.send(message, timeout=0.5)
time.sleep(0.01)
print('Sequence Stopped')




































