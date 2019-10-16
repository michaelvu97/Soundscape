#!/usr/bin/python

import numpy as np

#get Energy

#divide by expected distance distance

# sampleEnergies: an array of microphone energies <n x 1>
# sampleDistances: an array representing the distance from the source for each microphone <n x 1>
def getCalibrationFactors(sampleEnergies, sampleDistances):
	numEnergies = len(sampleEnergies)
	numDistances = len(sampleDistances)
	powers = {}
	if(numEnergies != numDistances):
		raise Exception("num energies(" + str(numEnergies) + ") != num distances(" + str(numDistances) + ")")

	for i in len(sampleEnergies):
		energy = sampleEnergies[i]
		distance = sampleDistances[i]
		powers.append(energy*(distance * distance))

	minPower = np.amin(powers);
	# (1/(x/minPower)) to scale the values around the minimum then find the relative strength 
	factors = [minPower/x for x in powers]

	return factors
