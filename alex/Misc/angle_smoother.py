# angle_smoother.py
import math

class AngleSmoother():
    def __init__(self):
        self.last_angle = 0.0
        self.last_last_angle = 0.0
        self.curr_angle = 0.0
        self.tolerance_degrees = 90
        self.hold_time = 0
        self.hold_time_threshold = 1
        self.weight1 = 0.15
        self.weight2 = 0.1

    def update(self, curr_angle_hypothesis):
    
        if(curr_angle_hypothesis == None):
            return None
        self.last_last_angle = self.last_angle
        self.last_angle = self.curr_angle
        self.curr_angle = curr_angle_hypothesis

        if self.last_angle is None or self.curr_angle is None:
            self.hold_time = 0 # Reset
            return

       # delta_ang = abs(self.last_angle - self.curr_angle) % 360
       # if delta_ang >= 180:
       #     delta_ang = 360 - delta_ang


        temp_curr = 0
        temp_last = self.last_angle - self.curr_angle
        temp_last_last = self.last_last_angle - self.curr_angle

        if(temp_last > 180):
            temp_last = temp_last - 360

        if(temp_last_last > 180):
            temp_last_last = temp_last_last - 360

        average = self.weight1*temp_last + self.weight2*temp_last_last
        average = average + self.curr_angle
        if(average < 0):
            average = average + 360

        guess = self.curr_angle 
        if(abs(average - self.last_last_angle) > 36):
            guess = self.curr_angle 


        self.curr_angle = average
        #print(average)
        #print(self.last_last_angle)
        #print()

        if 10 < self.tolerance_degrees:
            self.hold_time += 1
        else:
            self.hold_time = 0

    def getSmoothedAngle(self):
        return self.curr_angle # Hacked for now
        activated = self.hold_time >= self.hold_time_threshold
        if activated:
            return self.curr_angle
        else:
            return self.curr_angle

