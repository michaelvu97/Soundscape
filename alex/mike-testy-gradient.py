import numpy as np
import math 
import time

# Hypothesis functions (for testing)
f = [
    lambda t: np.array([math.cosh(t), math.sinh(t)]),
    lambda t: np.array([math.sinh(t) + 5, math.cosh(t)]),
    lambda t: np.array([-math.cosh(t) + 5.4, math.sinh(t)]),
]


K = len(f)

# Hypothesis function derivatives (for testing)
f_prime = [
    lambda t: np.array([math.sinh(t), math.cosh(t)]),
    lambda t: np.array([math.cosh(t), math.sinh(t)]),
    lambda t: np.array([-math.sinh(t), math.cosh(t)]),
]

# Parameterizations
t = np.array([0, 0, 0])

# Careful, if this is too big, it will explode.
learning_rate = 0.005

early_stopping_threshold = 0.001

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

