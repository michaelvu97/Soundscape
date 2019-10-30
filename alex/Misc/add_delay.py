import numpy as np

# Delays 
def delay(arr, tau):
    if tau == 0:
        return arr
    
    if tau > 0:
        return np.concatenate((np.zeros(tau), arr[:-tau]))

    # Negative delay
    test = (arr[-tau:], np.zeros(-tau))
    return np.concatenate(test)

