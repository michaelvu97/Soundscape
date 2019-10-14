import numpy as np
import math 

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

while True:
    curr_fprime = [x(t[i]) for i,x in enumerate(f_prime)]
    curr_hypotheses = [x(t[i]) for i,x in enumerate(f)]

    # For optimization
    # TODO: cache functional results.
    sum_point = np.sum(curr_hypotheses, axis=0)

    grad = lambda p, p_prime: np.dot((K + 1) * p - sum_point, p_prime)

    gradient = np.array([
        grad(curr_fprime[i], curr_hypotheses[i]) for i in range(K)
    ])

    # print(gradient)

    t = t - learning_rate * gradient;

    print("Predicted location: " + str(f[0](t[0])))