#Initialization
import random
import numpy as np
def ParameterInitialization(num_input, num_output):
    epsilon = 0.12
    #epsilon = 10000.0

    weight = np.zeros([num_output, num_input])
    for i in range(num_output):
        for j in range(num_input):
            weight[i][j] = random.uniform(-epsilon, epsilon)

    return weight
