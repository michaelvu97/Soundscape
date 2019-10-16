import numpy as np
import math 
import time
import matplotlib.pyplot as plt

speed_of_sound_mps = 330.0 # TODO the correct value.

"""
delta_x is the total horizontal distance between the mics
y_offset is the vertical distance from the origin.
delay_LR is the positive (or negative if flipped) amount of time that 
  microphone R lags behind L.
  e.g. if the sound source is very close to the left microphone, delay_LR > 0.
  Note that the sign of this delay is very important.

    returns: (f: t -> [x,y], f': t ->[x',y'])
"""
def generate_horizontal(delta_x, y_offset, delay_LR):
    delay_distance = -1 * speed_of_sound_mps * delay_LR
    a = 0.5 * delay_distance
    b = 0.5 * math.sqrt((delta_x ** 2) - (delay_distance ** 2))

    return (
        lambda t: np.array([
            a * math.cosh(t), 
            b * math.sinh(t) + y_offset
        ]),
        lambda t: np.array([
            a * math.sinh(t),
            b * math.cosh(t)
        ])
    )

"""
delta_y is the total vertical distance between the mics.
x_offset is the horizontal distance from the origin.
delay_TB is the positive (or negative if flipped) amount of time that
  microphone B (botom) lags behind T (top).
  e.g. if the sound souce is very close to the top microphone, delay_TD > 0.
  Note that the sign of the delay is very important.

  returns: (f: t -> [x,y], f': t ->[x',y'])
"""
def generate_vertical(delta_y, x_offset, delay_TD):
    delay_distance = speed_of_sound_mps * delay_TD
    a = 0.5 * delay_distance
    b = 0.5 * math.sqrt((delta_y ** 2) - (delay_distance ** 2))

    return (
        lambda t: np.array([
            b * math.sinh(t) + x_offset,
            a * math.cosh(t)
        ]),
        lambda t: np.array([
            b * math.cosh(t),
            a * math.sinh(t)
        ])
    )

"""
Generates a hyperbola rotated 45 degrees from horizontal. Must be centered 
   about the origin.
delta_pos is the distance between the two microphones.
delay_BL_TR is the amount of time that microphone TR (top right) lags behind
  BL (bottom_left). If the sound source is closer to BL, then delay_BL_TR > 0.

  returns: (f: t -> [x,y], f': t ->[x',y'])
"""
def generate_diagonal_BL_TR(delta_pos, delay_BL_TR):
    delay_distance = -1 * speed_of_sound_mps * delay_BL_TR
    a = 0.5 * delay_distance
    b = 0.5 * math.sqrt((delta_pos ** 2) - (delay_distance ** 2))

    rotation_coeff = 1/math.sqrt(2) # cos 45 = sin 45 = 1 / sqrt 2

    # Note that an arbitrary rotation can be achieved by creating a rotation
    # matrix and transforming the horizontal function by that amount.
    return (
        lambda t: np.array([
            rotation_coeff * (a * math.cosh(t) - b * math.sinh(t)),
            rotation_coeff * (a * math.cosh(t) + b * math.sinh(t))
        ]),
        lambda t: np.array([
            rotation_coeff * (a * math.sinh(t) - b * math.cosh(t)),
            rotation_coeff * (a * math.sinh(t) + b * math.cosh(t))
        ])
    )

"""
Generates a hyperbola rotated -45 degrees from horizontal. 
  Must be centered about the origin.
delta_pos is the distance between the two microphones.
delay_TL_BR is the amount of time that microphone BR (bottom right) lags behind
  TL (top left). If the sound source is closer to TL, then delay_BL_TR > 0.

  returns: (f: t -> [x,y], f': t ->[x',y'])
"""
def generate_diagonal_TL_BR(delta_pos, delay_TL_BR):
    delay_distance = -1 * speed_of_sound_mps * delay_TL_BR
    a = 0.5 * delay_distance
    b = 0.5 * math.sqrt((delta_pos ** 2) - (delay_distance ** 2))

    rotation_coeff = 1/math.sqrt(2) # cos 45 = sin 45 = 1 / sqrt 2
    # Note that an arbitrary rotation can be achieved by creating a rotation
    # matrix and transforming the horizontal function by that amount.

    return (
        lambda t: np.array([
            rotation_coeff * (a * math.cosh(t) + b * math.sinh(t)),
            rotation_coeff * (-a * math.cosh(t) + b * math.sinh(t))
        ]),
        lambda t: np.array([
            rotation_coeff * (a * math.sinh(t) + b * math.cosh(t)),
            rotation_coeff * (-a * math.sinh(t) + b * math.cosh(t))
        ])
    )

"""
Solves a system of estimated parametric equations.
functions: an array of tuples. Each tuple is (f(t), f'(t))
"""
def SolveEquations(functions, learning_rate = 0.005):
    
    K = len(functions)

    # Hypothesis functions (for testing)
    # TODO: replace with the generated lambdas.
    f = [x[0] for x in functions]

    # Hypothesis function derivatives (for testing)
    f_prime = [x[1] for x in functions]

    # Parameterizations
    t = np.array([0, 0, 0])

    # Careful, if this is too big, it will explode.
    learning_rate = 0.005

    def get_predicted_location():
        return np.mean([f[i](t[i]) for i in range(K)], axis=0)

    def get_error(curr_hypotheses):
        # TODO opt
        err = 0
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue

                err += np.sum(np.square(curr_hypotheses[i] - curr_hypotheses[j]))

        return err

    # TODO
    last_error = 1000000000
    min_error = last_error
    t1 = time.time()
    for i in range (500):
        curr_fprime = [x(t[i]) for i,x in enumerate(f_prime)]
        curr_hypotheses = [x(t[i]) for i,x in enumerate(f)]

        # For optimization
        # TODO: cache functional results.
        sum_point = np.sum(curr_hypotheses, axis=0)

        grad = lambda p, p_prime: np.dot((K + 1) * p - sum_point, p_prime)

        gradient = np.array([
            grad(curr_hypotheses[i], curr_fprime[i]) for i in range(K)
        ])

        # Update weights.
        t = t - learning_rate * gradient;

        curr_err = get_error(curr_hypotheses)
    t2 = time.time()

    print("Predicted location: " + str(get_predicted_location()))
    print("Elapsed time: " + str(t2 - t1) + " seconds")

    return get_predicted_location()

