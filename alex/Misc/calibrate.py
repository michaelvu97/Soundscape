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

y_intercept = 0;
slope = 1;
'''
got line of best fit from https://pythonprogramming.net/how-to-program-best-fit-line-machine-learning-tutorial/
'''
def compute_line_of_best_fit():
	slope = compute_slope()
	y_intercept = compute_y_intecept()

def compute_slope():
	return (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))

def compute_y_intercept(xs, ys):
 	return mean(ys) - mean(xs)

def estimate_energy(d):
	return slope * d + y_intercept

def compute_gain_coefficient(energy, previous_gain, distance):
	estimate = estimate_energy(distance);
	gain = previous_gain * math.sqrt(energy/ estimate)
	return gain;

def normalize_gains(gains):
	mean_gain = mean(gains)
	return [x / mean for x in gains]

def compute_normalized_gain_coefficients(energies, previous_gains, distances):
	gains = [];
	for i in range(0,n):
		gains.append(compute_gain_coefficient(energies[i], previous_gains[i], distances[i]))
	
	return normalize_gains(gains)

def compute_nth_gen_gain_coefficients(previous_gains, current_gains):
	new_gains = []

	for i in range(len(current_gains)):
		new_gains.append(
			current_gains[i] * LEARNING_RATE 
			+ previous_gains[i] * (1-LEARNING_RATE)
		)
	return new_gains;



if __name__ == "__main__":
	print("testing calibrate")
