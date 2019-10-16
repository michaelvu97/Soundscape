import numpy as np
import math 
import time
import matplotlib.pyplot as plt

speed_of_sound_mps = 330.0 # TODO the correct value.

"""
origin: [x,y] representing the origin of the hyperbola
dir1: [x, y] representing one of the directions of the hyperbolic asymptote,
    relative to the origin.
dir2: [x, y] representing one of the directions of the hyperbolic asymptote,
    relative to the origin.
"""
class HyperbolicAsymptote:
    def __init__(self, origin, dir1, dir2):
        self.origin = origin
        self.dir1 = dir1
        self.dir2 = dir2
    def __str__(self):
        return str(self.origin) + ": " + str(self.dir1) + ", " + str(self.dir2)

"""
delta_x is the total horizontal distance between the mics
y_offset is the vertical distance from the origin.
delay_LR is the positive (or negative if flipped) amount of time that 
  microphone R lags behind L.
  e.g. if the sound source is very close to the left microphone, delay_LR > 0.
  Note that the sign of this delay is very important.

    returns: HyperbolicAsymptote
"""
def generate_horizontal(delta_x, y_offset, delay_LR):
    delay_distance = -1 * speed_of_sound_mps * delay_LR
    a = 0.5 * delay_distance
    b = 0.5 * math.sqrt((delta_x ** 2) - (delay_distance ** 2))

    return HyperbolicAsymptote([0, y_offset], [a, b], [a, -b])

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

    return HyperbolicAsymptote([x_offset, 0], [b, a], [-b, a])

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
    return HyperbolicAsymptote(
        [0,0],
        [rotation_coeff * (a - b), rotation_coeff * (a + b)],
        [rotation_coeff * (a + b), rotation_coeff * (a- b)]
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
    return HyperbolicAsymptote(
        [0,0],
        [rotation_coeff * (a + b), rotation_coeff * (-a + b)],
        [rotation_coeff * (a - b), rotation_coeff * (-a - b)]
    )

def SolveEquationsIntersections(functions, granularity = 0.01):
    K = len(functions)
    f = [x[0] for x in functions]

    granularity_squared = granularity ** 2

    intersections = []

    t_samples = 5 * np.arcsinh(np.linspace(-3, 3, 500))

    for i in range(K - 1):
        for j in range(i + 1, K):
            # Compare function i and j

            # TODO: replace linspace with one that will be approx equidistant
            # for a hyperbola
            # TODO: this will limit the range
            for t_i in t_samples:
                for t_j in t_samples:
                    p_i = f[i](t_i)
                    p_j = f[j](t_j)
                    
                    # Use L2 distance
                    dist_squared = np.mean(np.square(p_i - p_j))
                    if dist_squared <= granularity_squared:
                        intersections.append(p_i)
                        break
    print(intersections)
    for intersection in intersections:
        plt.plot(intersection[0], intersection[1], 'bo')
    plt.show()
    # TODO find the closest intersections


"""
Solves a system of estimated parametric equations.
functions: an array of tuples. Each tuple is (f(t), f'(t))
"""
def SolveEquationsGradientDescent(functions, learning_rate = 0.005):
    
    K = len(functions)

    # Hypothesis functions
    f = [x[0] for x in functions]

    # Hypothesis function derivatives
    f_prime = [x[1] for x in functions]

    # Parameterizations
    t = np.zeros(K)

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
        t -= learning_rate * gradient
        print(get_predicted_location())

    t2 = time.time()

    print("Predicted location: " + str(get_predicted_location()))
    print("Elapsed time: " + str(t2 - t1) + " seconds")

    return get_predicted_location()

# Test where each mic is (1,1), (-1, 1) etc
# True point is at (4.3, 2.04)
D = 1
functions = [
    generate_horizontal(2, 1, -1.941 / speed_of_sound_mps),
    generate_vertical(2, 1, 1.0268 / speed_of_sound_mps),
    generate_diagonal_TL_BR(2 * math.sqrt(2), -0.9142 / speed_of_sound_mps),
    generate_horizontal(2, -1, -1.623133 / speed_of_sound_mps)
]
test_1 = np.array([functions[0][0](x) for x in np.linspace(-3, 3)])
test_2 = np.array([functions[1][0](x) for x in np.linspace(-3, 3)])
test_3 = np.array([functions[2][0](x) for x in np.linspace(-3, 3)])
test_4 = np.array([functions[3][0](x) for x in np.linspace(-3, 3)])

plt.plot(-1, -1, 'r*')
plt.plot(-1, 1, 'r*')
plt.plot(1, -1, 'r*')
plt.plot(1, 1, 'r*')

plt.plot(test_1[:,0], test_1[:,1])
plt.plot(test_2[:,0], test_2[:,1])
plt.plot(test_3[:,0], test_3[:,1])
plt.plot(test_4[:,0], test_4[:,1])

loc = SolveEquationsIntersections(functions, granularity = 0.01)

plt.plot(loc[0], loc[1], 'go')


plt.show()

