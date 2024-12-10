import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
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
		n = abs(int(dphi/(2*np.pi*speed*Ts)))
		t = np.arange(n)*Ts
		
		z = np.zeros(n)
		x = A*np.cos(-2*np.pi*speed*t + theta0)
		y = A*np.sin(-2*np.pi*speed*t + theta0)
	
	return [x,y,-z]
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



#code to get it to roll forward in y direction and back
Ts = 0.01
#roll forward
"""
[x1, y1, z1] = [round2half(i) for i in roll(4, 0, 0.1, [0, 0, 1], 20, Ts)]
[x2, y2, z2] = [round2half(i) for i in roll(4, 180, 0.1, [0, 0, 1], 20, Ts)]
"""
#rotate

[x1, y1, z1] = [round2half(i) for i in rotate(720, 1, 0.1, [1, 0, 0], 20, Ts)]
[x2, y2, z2] = [round2half(i) for i in rotate(720, -1, 0.1, [1, 0, 0], 20, Ts)]

#t = np.arange(2000)*Ts
#t1 = t[:1001]
#t2 = t[1001:]
"""
print(constant(t, 30).tolist())
x1 = [round2half(i) for i in constant(t, 30).tolist()]
x2 = [round2half(i) for i in constant(t, -30).tolist()]
y1 = [round2half(i) for i in constant(t, 0).tolist()]
y2 = [round2half(i) for i in constant(t, 0).tolist()]
z1 = [round2half(i) for i in constant(t, 0).tolist()]
z2 = [round2half(i) for i in constant(t, 0).tolist()]
"""
"""
x1 = constant(t1, 30).tolist()
x2 = constant(t2, -30).tolist()
y1 = constant(t1, 0).tolist()
y2 = constant(t2, 0).tolist()
z1 = constant(t1, 0).tolist()
z2 = constant(t2, 0).tolist()
"""
x =list(x1) + list(x2)
y =list(y1) + list(y2)
z =list(z1) + list(z2)
print(len(x))
print(len(y))
print(len(z))
#we are sampling at 10 ms
t = np.arange(len(x))*Ts


#t = np.arange(1000)*Ts

fileName = input("give file name: ")
current_dir = os.getcwd()
waveformFolder = os.path.join(current_dir, "waveforms")
waveformPlotFolder = os.path.join(current_dir, "waveformPlot")
waveformFilePath = os.path.join(waveformFolder, fileName + ".csv")
waveformPlotFilePath = os.path.join(waveformPlotFolder, fileName + ".png")
#plt.savefig(waveformPlotFilePath, format="png", bbox_inches="tight")
#plt.style.use('dark_background')

# Plot x vs t
plt.plot(t, x, label='I_x', color='blue', linewidth=2)

# Plot y vs t
plt.plot(t, y, label='I_y', color='red', linewidth=2)

# Plot z vs t
plt.plot(t, z, label='I_z', color='green', linewidth=2)

# Set title and axis labels
plt.title('Time Series Data - Current Over Time')
plt.xlabel('Time')
plt.ylabel('Current')

# Add legend
plt.legend()

# Show the plot
plt.show()


#d = {'t':t, 'f1':sin1, 'f2':sin2, 'f3':sin3, 'f4':sin4, 'f5':sin5, 'f6':sin6}
d = {'t':t, 'f1': x, 'f2':x, 'f3':y, 'f4':y, 'f5':z, 'f6':z}

df = pd.DataFrame(d)

#for linux
#df.to_csv("KreslingHelmholtz_CN/waveforms/" + + fileName + '.csv')
#for windows
df.to_csv(waveformFilePath)































