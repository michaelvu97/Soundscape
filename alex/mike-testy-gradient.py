import numpy as np
import math 

# Hypothesis functions (for testing)
f = [
    lambda t: (math.cosh(t), math.sinh(t)),
    lambda t: (math.sinh(t) + 5, math.cosh(t))
]

# Hypothesis function derivatives (for testing)
f_prime = [
    lambda t: (math.sinh(t), math.cosh(t)),
    lambda t: (math.cosh(t), math.sinh(t))
]

def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])

def mul(c, vec):
    return (c * vec[0], c * vec[1])

def dot(a,b):
    return a[0] * b[0] + a[1] * b[1]

# Parameterizations
t = [0, 0]

learning_rate = 0.0005


while True:
    curr_fprime = [x(t[i]) for i,x in enumerate(f_prime)]
    # This can be done with a neater sum in the future.
    curr_other = [
        sub(f[0](t[0]),f[1](t[1])),
        sub(f[1](t[1]),f[0](t[0]))
    ]

    gradient = [
        dot(curr_fprime[i], curr_other[i]) for i in range(2)
    ]

    print("grad: " + str(gradient))

    # TODO use numpy vectorized instead of this jank shit.
    t = sub(t, mul(learning_rate, gradient))

    print("t:" + str(t))