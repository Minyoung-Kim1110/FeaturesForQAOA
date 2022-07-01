"""
Author: Jaehwan Kim, Minyoung Kim ( June. 30, 2022)
2022 Hackaton 
Team: ThankQ
description: From graph, use qaoa to solve max-cut problem. 

Reference:
IBM Quantum. https://quantum-computing.ibm.com/, 2021
qiskit https://qiskit.org/textbook/ch-applications/qaoa.html

"""
from typing import List, Dict
import networkx as nx

from qiskit import QuantumCircuit, transpile
from qiskit.tools.monitor import job_monitor
from qiskit import Aer
from .graph import *

def cut(x: str, graph: nx.Graph):
    """
    Given a bitstring as a solution, this function returns
    the number of edges shared between the two partitions
    of the graph.
    
    Args:
        x: str
        solution bitstring
        
        graph: networkx graph
        
    Returns:
        obj: float
            Objective
    """
    cnt = 0.
    for i, j in graph.edges():
        if x[i] != x[j]:
            cnt += 1.

    return cnt


class Qaoa:
    def __init__(self, graph: nx.Graph, backend=None, backend_type: str = 'qasm_simulator', shots: int = 512):
        """_summary_

        Args:
            graph (nx.Graph): Graph to solve 
            backend (IBMQBackend, optional):  Defaults to None.
            backend_type (str, optional): simulator type . Defaults to 'qasm_simulator'.
            shots (int, optional): shots for backend. Defaults to 512.
        """
        self.graph = graph
        self.backend = backend if backend is not None else  Aer.get_backend(backend_type) 
        self.shots = shots

    def compute_expectation(self, counts: Dict[str, int]):
        """
        Computes expectation value based on measurement results
        - maxcut value 
        
        Args:
            counts: dict
                    key as bitstring, val as count
        Returns:
            avg: float
                expectation value
        """
        
        sum = 0
        sum_count = 0
        for bitstring, count in counts.items():
            sum += -cut(bitstring, self.graph) * count
            sum_count += count
            
        return sum / sum_count

    def create_qaoa_circ(self, betas: List[float], gammas: List[float]):
        """
        Creates a parametrized qaoa circuit
        For further explanation about QAOA, refer to https://qiskit.org/textbook/ch-applications/qaoa.html
        Args:  

            betas(List[float]): Parameters for problem unitary operator
            gammas(List[float]):Parameters for mixer unitary operator 
                        
        Returns:
            qc: qiskit circuit
        """
        assert(len(betas) == len(gammas))
        
        #Generate quantum circuit 
        nqubits = len(self.graph.nodes())
        qc = QuantumCircuit(nqubits)
        
        # Set initial_state with hadamard gate
        for i in range(0, nqubits):
            qc.h(i)
        
        for beta, gamma in zip(betas, gammas):
            # problem unitary
            for i, j in list(self.graph.edges()):
                qc.rzz(2 * beta, i, j)

            # mixer unitary
            for i in range(0, nqubits):
                qc.rx(2 * gamma, i)
        #Measure
        qc.measure_all()
        return qc

    # Finally we write a function that executes the circuit on the chosen backend
    def execute_circ(self, params: List[float]):
        betas, gammas = params[:len(params)//2], params[len(params)//2:]
        
        qc = self.create_qaoa_circ(betas, gammas)
        counts = self.backend.run(qc, seed_simulator=10, nshots=self.shots).result().get_counts()
        # transpiled_circuit = transpile(qc, self.backend, optimization_level=3)
        # job = self.backend.run(transpiled_circuit)
        # job_monitor(job, interval=2)

        # results = job.result()
        # counts = results.get_counts()
        
        return self.compute_expectation(counts)

if __name__ == "__main__":
    #import minimize module
    from scipy.optimize import minimize


    def get_example_graph():
        """generate sample graph 

        Returns:
            graph(nx.Graph): generated graph with 4 nodes and 4 edges with edge weight = 1 
        """
        graph = nx.Graph()
        graph.add_nodes_from([0, 1, 2, 3])
        graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
        nx.draw(graph, with_labels=True, alpha=0.8, node_size=500)
        return graph

    
    def get_graph_from_data(filepath:str, node_num:int = 4  , dist_ftn = lambda x, y : np.sqrt(np.dot(x-y, x-y))):
        """generate sample graph from given data 

        Args:
            filepath (str): _description_
            node_num (int, optional): number of data points (node of graph). Defaults to 4.
            dist_ftn (ftn, optional): distance function.

        Returns:
            graph(nx.Graph): generated graph with given node_num and edge weight is calculated by distance function
        """
        data = np.load(filepath)
        data = data[:node_num, 1:]
        graph = generate_graph(data, dist_ftn)
        return graph
    
    filepath = "../data/drug145_5.npy"
    
    graph = get_example_graph()
    #params length = 2p [betas, gammas]
    params = [1.0, 1.0]
    qaoa = Qaoa(graph)

    res = minimize(qaoa.execute_circ, 
                        params, 
                        method='COBYLA')
    print(res)