import numpy as np

cosftn = lambda x, y : np.dot(x, y)/ norm(x)/norm(y)
Euclidean = lambda x, y : np.sqrt(np.dot(x-y, x-y))

#Euclidean distance 
def norm(x):
    return np.sqrt(np.dot(x, x))

def cos_dist(x, y):
    #actually, negative cos function 
    return -np.dot(x, y)/ norm(x)/norm(y)

def L2(x, y):
    #Euclidean distance
    return norm(x-y)

def L1(x, y):
    #Manhattan distance 
    return np.sum(np.abs(x-y))

distance_func_list = [ cos_dist, L2, L1 ]

class QuantumKernelMap():
    def __init__(self, backend):
        self.backend = backend

    def run(data:np.array):
        num_datapt, num_features = data.shape
        feature_map =ZZFeatureMap(feature_dimension=num_features, reps=2, entanglement="linear")
        kernel = QuantumKernel(feature_map=feature_map, quantum_instance=self.backend)
        return kernel.evaluate(data)
