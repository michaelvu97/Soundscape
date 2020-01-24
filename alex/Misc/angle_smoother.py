# angle_smoother.py
import math

class AngleSmoother():
    def __init__(self):
        self.last_angle = None
        self.curr_angle = None
        self.tolerance_degrees = 90
        self.hold_time = 0
        self.hold_time_threshold = 2

    def update(self, curr_angle_hypothesis):
        self.last_angle = self.curr_angle
        self.curr_angle = curr_angle_hypothesis

        if self.last_angle is None or self.curr_angle is None:
            self.hold_time = 0 # Reset
            return

        delta_ang = abs(self.last_angle - self.curr_angle) % 360
        if delta_ang >= 180:
            delta_ang = 360 - delta_ang

        if delta_ang < self.tolerance_degrees:
            self.hold_time += 1
        else:
            self.hold_time = 0

    def getSmoothedAngle(self):
        return self.curr_angle # Hacked for now
        activated = self.hold_time >= self.hold_time_threshold
        if activated:
            return self.curr_angle
        else:
            return None

