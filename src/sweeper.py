''' 
for various distance functions,
execute QAOA 
and compare results
''' 
from QAOA import *
from graph import * 
from distance import *
from scipy.optimize import minimize


# 노드 개수 
data_num = 4 
#p 개수 
p_list = [1,2]

#data import 
data = np.load("../data/drug145_5.npy")
data = data[:data_num, 1:]

#Sweep! 
for func in distance_func_list:
    graph = generate_graph(data, func)
    for p in p_list: 
        #generate beta and gammas 
        print(f"p value is {p} and using function {func}")
        params = [1.0 for i in range(2*p)]
        qaoa = Qaoa(graph)

        res = minimize(qaoa.execute_circ, 
                        params, 
                        method='COBYLA')
        print(res.fun)
        print()


