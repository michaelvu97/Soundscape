import numpy as np

class LiveCalibration():



    def __init__(self, num_channels):
        self.base_rate = 0.0005 # The weighting of how much to adapt to new energy levels.
        self.num_channels = num_channels
        self.gain = np.ones((num_channels, 1), dtype=np.float32)

    def rate(self):
        if self.base_rate == 0:
            return 0

        # self.base_rate -= 0.0000001
        # if (self.base_rate < 0):
        #     self.base_rate = 0
        return self.base_rate

    """
    returns an np array of the gain coefficients for each channel, for the
    following property to hold true G[0] * N[0] = G[1] * N[1] = ...
    Essentially, the same signal will be the same energy level after passing 
    through the microphone system.
    """
    def get_gain_correction(self, chunk_energy):
        rate = self.rate()
        if rate == 0:
            # When it's done calibrating.
            return self.gain

        chunk_energy_post_gain = chunk_energy * self.gain

        avg_post_gain_energy = np.mean(chunk_energy_post_gain)

        target_gain = avg_post_gain_energy / chunk_energy

        # print("current_gain=" + str(self.gain) + " target_gain=" + str(target_gain))

        self.gain += (target_gain - self.gain) * rate

        
        # Normalize
        self.gain = self.gain - np.min(self.gain) + 1.0
        return self.gain



