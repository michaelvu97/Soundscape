import numpy as np
import math 
import time
import matplotlib.pyplot as plt

speed_of_sound_mps = 330.0 # TODO the correct value.
slice_size = 22.5;

"""
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
represents a line (y = mx + b)
origin: [x,y] representing the origin of the hyperbola
direction: [x, y] representing one of the directions of the hyperbolic asymptote,
    relative to the origin.
"""
class Line:
    def __init__(self, origin, direction):
        self.origin = origin
        self.x = direction[0]
        self.y = direction[1]
        self.theta = None;
    def __str__(self):
        return "theta: " + str(self.theta)  + " section: " + str(self.getDirectionSection()) 
    
    #assuming the line intersects the origin even though it doesn't in most cases
    #returns theta E [0, 2*pi)
    def getTheta(self):
        if(self.theta != None):
            return self.theta;

        theta =  math.atan2(self.y,self.x)
        if(theta < 0):
            theta = 2 * math.pi + theta
        theta = theta * 180 / math.pi
        self.theta = theta;
        return theta
    
    #Assuming a circling is dividing into 16 equally sized slices, return which slice number this line belongs to
    #returns [0, 15]
    def getDirectionSection(self):
        theta = self.getTheta()
        return math.floor(theta /slice_size)
    
'''
returns theta between [-pi, pi] given the x and y of a triangle
'''
def atan2(x,y):
    theta = 0;
    if(x == 0):
        if(y == 0):
            raise Exception("direction of line is 0,0")
        elif(y > 0):
            theta = math.pi/2;
        else:
            theta = -math.pi/2;
    elif(x > 0):
        theta = np.arctan(y/x);
    elif(x < 0):
        if(y <0):
            theta = np.arctan(y/x) - math.pi
        else:
            theta = np.arctan(y/x) +  math.pi

    return theta
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
    b = 0.5 * math.sqrt(max(0, (delta_x ** 2) - (delay_distance ** 2)))

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
    b = 0.5 * math.sqrt(max(0, (delta_y ** 2) - (delay_distance ** 2)))

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
    b = 0.5 * math.sqrt(max(0, (delta_pos ** 2) - (delay_distance ** 2)))

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
    b = 0.5 * math.sqrt(max(0, (delta_pos ** 2) - (delay_distance ** 2)))

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




def addToSectionMap(sectionMap, origin, direction):
    line = Line(origin, direction)
    theta = line.getTheta()
    print(line)
    section = line.getDirectionSection()
    if((section in sectionMap) == False):
        sectionMap[section] = []
    sectionMap[section].append(line)


def getDirectionGivenAsymptotes(asymptotes):
    sectionMap = {} 
    for asymptote in asymptotes:
        addToSectionMap(sectionMap, asymptote.origin, asymptote.dir1)   
        addToSectionMap(sectionMap, asymptote.origin, asymptote.dir2)   
    
    
    mostLines = 0;
    lineArr = [];
    for lines in sectionMap.values():
        numLines = len(lines)
        if(numLines > mostLines):
            mostLines = numLines;
            lineArr = lines
    
    averageTheta = 0; 
    
    for line in lineArr:
        averageTheta += line.getTheta();
    
    averageTheta = averageTheta / mostLines;
    print("average angle of lines in the section with the most lines (" + str(mostLines) + ") is " + str(averageTheta))

def normalize(vec):
    length = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
    return [vec[0] / length, vec[1] / length]

def getDirectionGivenAsymptotesNew(asymptotes):
    tolerance_degrees = 10

    normalized_vectors_temp = [[normalize(a.dir1), normalize(a.dir2)] for a in asymptotes]
    normalized_vectors = []
    for v in normalized_vectors_temp:
        normalized_vectors.append(v[0])
        normalized_vectors.append(v[1])

    # Cluster points
    angle_tolerance_rads = math.radians(22.5)
    cluster_tolerance_dist_squared = 2 * (1 - math.cos(angle_tolerance_rads))

    print("tolerance: " + str(cluster_tolerance_dist_squared))
    clusters = [] # form: (average point, num_points)
    for p in normalized_vectors:
        cluster_found = False
        for cluster in clusters:
            cluster_av = cluster[0]
            if (((p[0] - cluster_av[0]) **2) + ((p[1] - cluster_av[1]) ** 2) <= cluster_tolerance_dist_squared):
                cluster_found = True
                cluster[1] += 1
                # Update average
                cluster[0] = [(p[0] + cluster_av[0]) / (cluster[1]), (p[1] + cluster_av[1]) / (cluster[1])]
                break
        if not cluster_found:
            clusters.append([p, 1])

    m_val = 0
    m_dir = clusters[0][0]
    for c in clusters:
        if c[1] > m_val:
            m_dir = c[0]
            m_val = c[1]

    # Convert the average direction to an angle
    theta_rads = math.atan2(m_dir[1], m_dir[0])
    if (theta_rads < 0):
        theta_rads += math.radians(360)

    return math.degrees(theta_rads)

if __name__ =="__main__":
    # Test where each mic is (1,1), (-1, 1) etc
    # True point is at (4.3, 2.04)
    D = 1
    asymptotes = [
        generate_horizontal(2, 1, -1.941 / speed_of_sound_mps),
        generate_vertical(2, 1, 1.0268 / speed_of_sound_mps),
        generate_diagonal_TL_BR(2 * math.sqrt(2), -0.9142 / speed_of_sound_mps),
        generate_horizontal(2, -1, -1.623133 / speed_of_sound_mps)
    ]
    
    getDirectionGivenAsymptotes(asymptotes)

