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
	
	

print("hello world")

bus = can.interface.Bus(bustype='socketcan', channel='can0', bitrate=1000000)
print("Canbus Successfully Setup. \n")



values = [0,0,0,0,0,0]
tx = encode(values)
message = can.Message(arbitration_id=0x00, is_extended_id=False, data= tx)
bus.send(message, timeout=0.5)
time.sleep(0.01)

df = pd.read_csv("fwdbwd50.csv")
samples1 = list(df['f1'])
samples2 = list(df['f2'])
samples3 = list(df['f3'])
samples4 = list(df['f4'])
samples5 = list(df['f5'])
samples6 = list(df['f6'])
print("Input Data Successfully Loaded. \n")




while True:
	signal = input("Enter 'a' to start Sequence:")
	if (signal == 'a'):
		print("Sequence Started")
		break;
	

resetToZero = True
for i in range(len(samples1)):
	values = [samples1[i], samples2[i], samples3[i], samples4[i], samples5[i], samples6[i]]
	if samples1[i] == 0.0 and resetToZero:
		print("EndingCurrent")
		resetToZero = False
		
	tx = encode(values)
	message = can.Message(arbitration_id=0x00, is_extended_id=False, data= tx)
	bus.send(message, timeout=0.5)
	time.sleep(0.01)

values = [0,0,0,0,0,0]
tx = encode(values)
message = can.Message(arbitration_id=0x00, is_extended_id=False, data= tx)
bus.send(message, timeout=0.5)
time.sleep(0.01)
print("Finished")




































