#!/usr/bin/python

import numpy as np
from add_delay import delay;

#find the number of indices you need to delay b by so it matches up with a
def findDelay(a,b):
	maxEnergy = 0;
	indexDelay = 0;
	energies = {};
	for i in range(-len(a), len(a)):
		delay(b,i)
		corr = np.correlate(a, delay(b, i), mode='valid')
		energy = np.sum(corr)
		energies[i] = energy

		if(energy > maxEnergy):
			indexDelay = i
			maxEnergy = energy
	
	return indexDelay



# with respect to the first signal, find how many indices behind it is with respect to the others
def findDelaysOfSignals(signals):
	if(len(signals) == 0):
		raise Exception("called synchronize with empty array")

	delays = {};
	reference = signals[0]
	for i, signal in enumerate(signals):
		if(len(signal) == 0):
			raise Exception("signal " + str(i) + " is empty")
		
		delays[i] = findDelay(reference, signal);

	return delays



if __name__ == "__main__":
	testSignals = [
		[1,2,1,0,0],
		[0,1,2,1,0],
		[0,0,1,2,1]
	]

	delays = findDelaysOfSignals(testSignals);

	if(np.array_equal(delays, [1, -1, -2])):
		raise Exception("TEST FAILED")

	else:
		print("test passed");
