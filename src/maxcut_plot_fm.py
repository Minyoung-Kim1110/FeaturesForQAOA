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
from QAOA_copy import *
from graph import * 
from distance import *
from scipy.optimize import minimize
from qiskit.test.mock import FakeProvider


# Loading your IBM Quantum account(s)
# IBMQ.save_account('459bae2c9f5515389b12ccadc82836b5872efac3efc856c982715acee8e3ccaa9b10d04704743128f3c9155fec9ed64b8f92627bc3928e4b343b0425e926af2c')
IBMQ.load_account()

# fake_provider = FakeProvider()
# print(fake_provider.backends())
# fake_backend = fake_provider.get_backend('fake_montreal')

provider = IBMQ.get_provider(hub='ibm-q-skku', group='snu', project='snu-graduate')
# backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 7 and
#                                     not x.configuration().simulator and x.status().operational==True))
backend = provider.get_backend('ibmq_kolkata')
print("least busy backend: ", backend)

data = np.load("../data/drug145_5.npy")
xdata = data[:, 1:]

# 노드 개수 
data_num = 3
#p 개수 
p_list = [1]

#Sweep! 
filepath = './result_fm.txt'
# kernel = QuantumKernelMap(backend)
# kernel.run(xdata[:data_num])
mat = np.load("mat.npy")
# mat = np.random.random((3,3))
np.fill_diagonal(mat, 0)
graph = generate_graph_from_numpy(mat, draw=False)
for p in p_list: 
    #generate beta and gammas 
    print(f"p value is {p} and using function QuantumKernelMap")
    params = [1.0 for i in range(2*p)]
    qaoa = Qaoa(graph=graph, backend=backend)

    res = minimize(qaoa.execute_circ, 
                    params, 
                    method='COBYLA')
    print(res.fun)
    print()
    with open(filepath, "a") as file: 
        file.write(f"data number={data_num} p = {p}, distance function = QuantumKernelMap")
        file.write("\n")
        file.write(str(res))
        file.write("\n========================================================================================================\n")
    