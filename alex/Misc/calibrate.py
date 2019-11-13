#!/usr/bin/python

'''
using
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Tashev_MicArraySelfCalibration_ICME_04.pdf
to calibrate gains of microphones
'''
import numpy as np
from statistics import mean
import math

LEARNING_RATE = 0.001;

'''
got line of best fit from https://pythonprogramming.net/how-to-program-best-fit-line-machine-learning-tutorial/
'''
def compute_line_of_best_fit(xs, ys):
	slope = compute_slope(xs, ys)
	y_intercept = compute_y_intercept(xs, ys)
	
	return [slope, y_intercept]

def compute_slope(xs, ys):

	return (((mean(xs)*mean(ys)) - mean(xs*ys)) / ((mean(xs)*mean(xs)) - mean(xs*xs)))

def compute_y_intercept(xs, ys):
	return mean(ys) - mean(xs)


def compute_gain_coefficient(distance, energy, previous_gain, slope, y_intercept):
	estimate = distance * slope +  y_intercept;
	
	gain = previous_gain * math.sqrt(energy/ estimate)
	return gain;

def normalize_gains(gains):
	mean_gain = mean(gains)
	return [x / mean_gain for x in gains]

def compute_normalized_gain_coefficients(distances, energies, previous_gains):
	slope, y_intercept = compute_line_of_best_fit(distances, energies);
	gains = [];
	
	for i in range(len(previous_gains)):
		gain = compute_gain_coefficient(distances[i], energies[i], previous_gains[i], slope, y_intercept)
		gains.append(gain)
	
	return normalize_gains(gains)

def compute_next_gen_gain_coefficients(distances, energies, previous_gains):
	current_gains = compute_normalized_gain_coefficients(distances, energies, previous_gains)
	new_gains = []

	for i in range(len(current_gains)):
		new_gains.append(
			current_gains[i] * LEARNING_RATE 
			+ previous_gains[i] * (1-LEARNING_RATE)
		)
	return new_gains;


'''
energies - 2d array of energies where each row contain the energy values for the n microphones for some distance
location - 2d array of energies where each row represents the actual distance from the sound source for a mic
'''
def compute_gains(energies_2d, distances_2d):
	num_energies = len(energies_2d)
	assert(num_energies == len(distances_2d))
	assert(num_energies > 0)
	num_mics = len(energies_2d[0])
	gains = np.ones((num_mics));

	for i in range(len(energies_2d)):
		energies = energies_2d[i]
		distances = distances_2d[i]
		num_energies = len(energies)
		
		assert(num_energies == len(distances) and num_energies == num_mics)
		
		gains = compute_next_gen_gain_coefficients(distances, energies, gains)

if __name__ == "__main__":
	print("testing calibrate")
	
	'''tests'''
	test1 = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
	
	
	'''compute line tests'''
	res = compute_line_of_best_fit(test1, test1);
	assert(res[1] == 0)
	assert(res[0] == 1)

	test2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
	
	res = compute_line_of_best_fit(test1, test2);
	assert(res[0] == 1)
	assert(res[1] == 1)
	
	test3 = np.array([2,1,4,3,6])
	res = compute_line_of_best_fit(test1, test3);
	print(res)
	
	'''normalize gain tests'''
	gains = normalize_gains(test1);
	for i in range(len(gains)):
		assert(gains[i] == test1[i]/2.0)

	gains = normalize_gains(test2);
	for i in range(len(gains)):
		assert(gains[i] == test2[i]/3.0)

	'''compute coefficient tests'''
	expected_gains = np.array([1,math.sqrt(0.5), 2* math.sqrt(0.5)]); 
	
	inputs = np.array([
		[1.0, 1.0, 1.0, 1.0, 0],
		[1.0, 1.0, 1.0, 1.0, 1.0],
		[1.0, 1.0, 2.0, 1.0, 1.0]
	])

	for i in range(len(inputs)):
		assert(compute_gain_coefficient(*inputs[i]) == expected_gains[i])
	
	'''compute coefficients test'''
	
	'''expect no change'''
	expected_gains2 = [1, 1, 1, 1, 1]
	inputs = np.array([ test1, test2, [1, 1, 1, 1, 1] ])
	gains = compute_normalized_gain_coefficients(*inputs)	
	for i in range(len(gains)):
		assert(gains[i] == expected_gains2[i])
	

	res = compute_line_of_best_fit(test1, test3)
	
	inputs = np.array([test2, test3, [1, 1, 1, 1, 1]])
	#print(inputs)
	gains = compute_normalized_gain_coefficients(*inputs)	
	#print(gains)

	print("tests passed")
		
