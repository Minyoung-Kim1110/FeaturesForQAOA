"""
Author: Hanseo Sohn, Minyoung Kim (Jul. 1, 2022)
2022 Hackaton 
Team: ThankQ
description: 
    for various parameters(data point number, p, etc...) execute qaoa with fake backend and real backend 
    Save data in results folder
    results from fake backend is saved in '../result/max_cut_fake_backend.txt'
    results from real backend is saved in '../result/max_cut_{backend}.txt' 

Execute: 
    in the src folder, use below command 
    python maxcut_plot.py false
or  python maxcut_plot.py true

Reference:
IBM Quantum. https://quantum-computing.ibm.com/, 2021

"""
import numpy as np

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit.visualization import *
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.ibmq import least_busy
from qiskit import assemble
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
from .QAOA import *
from .graph import * 
from .distance import *
from scipy.optimize import minimize
from qiskit.test.mock import FakeProvider



#Get the results from simulator 
def run_simulator():
    data_num = 4
    p_list = np.arange(10)
    #Set fake provider 
    IBMQ.load_account()
    fake_provider = FakeProvider()
    backend = fake_provider.get_backend('fake_montreal')
    filepath = '../result/max_cut_fake_backend.txt'
    run_qaoa()
    
def run_real_backend():
    data_num = 4 
    p_list = np.arange(4)
    IBMQ.save_account('459bae2c9f5515389b12ccadc82836b5872efac3efc856c982715acee8e3ccaa9b10d04704743128f3c9155fec9ed64b8f92627bc3928e4b343b0425e926af2c')
    provider = IBMQ.get_provider(hub='ibm-q-skku', group='snu', project='snu-graduate')
    backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 7 and
                                    not x.configuration().simulator and x.status().operational==True))
    print("least busy backend: ", backend)
    filepath = '../result/max_cut_'+str(backend)+'.txt'
    run_qaoa()    
        
def save_data():
    with open(filepath, "a") as file: 
        file.write(f"data number={data_num} p = {p}, distance function = {func.__name__}")
        file.write("\n")
        file.write(str(res))
        file.write("\n========================================================================================================\n")

def run_qaoa():
    #Run QAOA with distance functions in original space
    for func in distance_func_list: 
        graph = generate_graph(data[:data_num], func, draw=False)
        for p in p_list: 
            print(f"p value is {p} and using function {func.__name__}")
            params = [1.0 for _ in range(2*p)]
            qaoa = Qaoa(graph=graph, backend=backend)
            res = minimize(qaoa.execute_circ, 
                                params, 
                                method='COBYLA')
            print("Finish minimize")
            print(res.fun)
            save_data()
    #Run QAOA in feature space 
    #Check distance matrix is generated 
    from os.path import exists 
    file_exists = exists("../results/adj_mat.npy")
    if file_exists:
        #If file exists, use that file 
        mat = np.load("../results/adj_mat.npy")
    else: 
        #else, use quantum kernel map to calculate distance
        kernel = QuantumKernelMap(backend)
        mat = kernel.get_distance(data[:data_num])
        np.save("../results/adj_mat.npy", mat)
    graph = generate_graph_from_numpy(mat, draw=False)
    for p in p_list: 
        print(f"p value is {p} and using function QuantumKernelMap")
        params = [1.0 for _ in range(2*p)]
        qaoa = Qaoa(graph=graph, backend=backend)
        res = minimize(qaoa.execute_circ, 
                            params, 
                            method='COBYLA')
        print("Finish minimize")
        print(res.fun)
        save_data()

    

if __name__ == "__main__":
    import sys 
    real = False 
    if sys.argv[1] is not None: 
        real = sys.argv[1]
    #Load data 
    data = np.load("../data/drug145_5.npy")
    #Erase label 
    data = data[:, 1:]
    

    #execute
    if real: 
        run_real_backend()
    else: 
        run_fake_backend()







