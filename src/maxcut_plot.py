import numpy as np

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.providers.aer import QasmSimulator

from qiskit.providers.ibmq import least_busy
from qiskit import assemble
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
from QAOA import *
from graph import * 
from distance import *
from scipy.optimize import minimize
from qiskit.test.mock import FakeProvider


# # Loading your IBM Quantum account(s)
# # IBMQ.save_account('459bae2c9f5515389b12ccadc82836b5872efac3efc856c982715acee8e3ccaa9b10d04704743128f3c9155fec9ed64b8f92627bc3928e4b343b0425e926af2c')
# IBMQ.load_account()

# fake_provider = FakeProvider()
# # print(fake_provider.backends())
# fake_backend = fake_provider.get_backend('fake_montreal')

# # provider = IBMQ.get_provider(hub='ibm-q-skku', group='snu', project='snu-graduate')
# # backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 7 and
# #                                     not x.configuration().simulator and x.status().operational==True))
# # print("least busy backend: ", backend)

# data = np.load("../data/drug145_5.npy")
# xdata = data[:, 1:]
# #Sweep! 
# filepath = '../result/max_cut_fake_backend.txt'
def run():
    for dn in data_num:
        for func in distance_func_list:
            graph = generate_graph(xdata[:dn], func, draw=False)
            for p in p_list: 
                #generate beta and gammas 
                print(f"p value is {p} and using function {func.__name__}")
                params = [1.0 for i in range(2*p)]
                qaoa = Qaoa(graph=graph, backend=fake_backend)

                res = minimize(qaoa.execute_circ, 
                                params, 
                                method='COBYLA')
                print(res.fun)
                print()
                with open(filepath, "a") as file: 
                    file.write(f"data number={dn} p = {p}, distance function = {func.__name__}")
                    file.write("\n")
                    file.write(str(res))
                    file.write("\n========================================================================================================\n")


if __name__ == "__main__":
    import sys 
    real = False 
    if sys.argv[1] is not None: 
        real = sys.argv[1]
    if real: 
        IBMQ.save_account('459bae2c9f5515389b12ccadc82836b5872efac3efc856c982715acee8e3ccaa9b10d04704743128f3c9155fec9ed64b8f92627bc3928e4b343b0425e926af2c')
        provider = IBMQ.get_provider(hub='ibm-q-skku', group='snu', project='snu-graduate')
        backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 7 and
                                    not x.configuration().simulator and x.status().operational==True))
        print("least busy backend: ", backend)
        filepath = '../result/max_cut_'+str(backend)+'.txt'

    else: 
        IBMQ.load_account()
        fake_provider = FakeProvider()
        fake_backend = fake_provider.get_backend('fake_montreal')
        filepath = '../result/max_cut_fake_backend.txt'

    
    #Load data 
    data = np.load("../data/drug145_5.npy")
    #Erase label 
    data = data[:, 1:]
    #Set node number and p     
    data_num = list(range(5, 10)) 
    p_list = list(range(1, 5))
    

    #Execute 
    run()







