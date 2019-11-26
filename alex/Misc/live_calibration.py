import numpy as np

class LiveCalibration():



    def __init__(self, num_channels):
        self.base_rate = 0.02 # The weighting of how much to adapt to new energy levels.
        self.decay_rate = 0.001
        self.num_channels = num_channels
        self.avg_energy_level = None
        self.has_any_data = False


    def rate(self):
        if self.base_rate == 0:
            return 0

        self.base_rate -= 0.0001
        if (self.base_rate < 0):
            self.base_rate = 0
        return self.base_rate

    def update(self, chunk_energy):
        if self.avg_energy_level is None:
            self.avg_energy_level = chunk_energy
            return 

        rate = self.rate()
        if rate == 0:
            return

        print("CALIBRATING")

        self.avg_energy_level = rate * np.array(chunk_energy) + (1.0 - rate) * self.avg_energy_level

    """
    returns an np array of the gain coefficients for each channel, for the
    following property to hold true G[0] * N[0] = G[1] * N[1] = ...
    Essentially, the same signal will be the same energy level after passing 
    through the microphone system.
    """
    def get_gain_correction(self):
        max_energy = max(self.avg_energy_level)
        return max_energy / self.avg_energy_level





