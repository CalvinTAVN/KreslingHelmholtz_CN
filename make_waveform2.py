import spidev
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def roll(step_no, angle, speed, init, A, Ts):
	k = int(1/(4*speed*Ts))  #number of samples per quarter roll
	[x0, y0, z0] = init #initial magnetization orientation
	
	theta = angle*np.pi / 180
	Ax = A*np.cos(theta)
	Ay = A*np.sin(theta)
	
	n = k*step_no
	t = np.arange(n)*Ts
	
	
	x_state = 0
	y_state = 0
	z_state = 0

	if (z0 == 1 or z0 == -1):
		if (z0%4 == 0):
			z_state = z0
		elif (z0%4 == 2):
			z_state = - z0 
	
	if (z0 == 1):  #if magnetization points up, the x0,y0 are irrelevant.
		z = A*np.cos(2*np.pi*speed*t)
		x = Ax*np.sin(2*np.pi*speed*t)
		y = Ay*np.sin(2*np.pi*speed*t)
		return [x,y,-z]
	elif (z0 == -1):  #if magnetization points down, the x0,y0 are irrelevant.
		z = -A*np.cos(2*np.pi*speed*t)
		x = -Ax*np.sin(2*np.pi*speed*t)
		y = -Ay*np.sin(2*np.pi*speed*t)
		return [x,y,-z]
	
	
	
	



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

t = np.arange(1000)*Ts
first_Period = t[:750]
second_Period = t[750:]
x1 = [round(2*i)/2 for i in constant(first_Period, 40)]
x2 = [round(2*i)/2 for i in constant(second_Period, -50)]
x1.extend(x2)
x = x1
y = [round(2*i)/2 for i in constant(t, 0)]
z = [round(2*i)/2 for i in constant(t, 0)]


#to make it roll in diagonal direction
"""
[x1, y1, z1] = [round2half(i) for i in roll(5, 135, 0.1, [0,0,1], 30, Ts)]
[x2, y2, z2] = [round2half(i) for i in roll(5, 20, 0.1, [0,0,-1], 30, Ts)]


x1 = round2half(x1)
y1 = round2half(y1)
z1 = round2half(z1)

x = list(x1) + list(x2)
y = list(y1) + list(y2)
z = list(z1) + list(z2)

t = np.arange(len(x))*Ts
"""
plt.plot(t, x)
plt.plot(t, y)
plt.plot(t, z)
plt.show()

#d = {'t':t, 'f1':sin1, 'f2':sin2, 'f3':sin3, 'f4':sin4, 'f5':sin5, 'f6':sin6}
d = {'t':t, 'f1': x, 'f2':x, 'f3':y, 'f4':y, 'f5':z, 'f6':z}

df = pd.DataFrame(d)
df.to_csv("fwdbwd50.csv")
































