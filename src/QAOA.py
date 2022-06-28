'''
From graph, 
use QAOA tutorial, calculate results 
output: classification label
''' 
from typing import List, Dict
import networkx as nx

from qiskit import QuantumCircuit
from qiskit import Aer
from graph import *

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
    cnt = 0
    for i, j in graph.edges():
        if x[i] != x[j]:
            cnt += 1

    return cnt


class Qaoa:
    def __init__(self, graph: nx.Graph, backend_type: str = 'qasm_simulator', shots: int = 512):
        self.graph = graph
        self.backend = Aer.get_backend(backend_type)
        self.backend.shots = shots
        self.shots = shots

    def compute_expectation(self, counts: Dict[str, int]):
        """
        Computes expectation value based on measurement results
        
        Args:
            counts: dict
                    key as bitstring, val as count
            
            graph: networkx graph
            
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
        
        Args:  
            graph: networkx graph
            theta: list
                unitary parameters
                        
        Returns:
            qc: qiskit circuit
        """
        assert(len(betas) == len(gammas))
        
        nqubits = len(self.graph.nodes())
        qc = QuantumCircuit(nqubits)
        
        # initial_state
        for i in range(0, nqubits):
            qc.h(i)
        
        for beta, gamma in zip(betas, gammas):
            # problem unitary
            for i, j in list(self.graph.edges()):
                qc.rzz(2 * beta, i, j)

            # mixer unitary
            for i in range(0, nqubits):
                qc.rx(2 * gamma, i)

        qc.measure_all()

        return qc

    # Finally we write a function that executes the circuit on the chosen backend
    def execute_circ(self, params: List[float]):
        betas, gammas = params[:len(params)//2], params[len(params)//2:]
        
        qc = self.create_qaoa_circ(betas, gammas)
        counts = self.backend.run(qc, seed_simulator=10, 
                            nshots=self.shots).result().get_counts()
        
        return self.compute_expectation(counts)

if __name__ == "__main__":
    def get_example_graph():
        graph = nx.Graph()
        graph.add_nodes_from([0, 1, 2, 3])
        graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
        nx.draw(graph, with_labels=True, alpha=0.8, node_size=500)
        return graph

    from scipy.optimize import minimize

    data = np.load("../data/drug145_5.npy")
    data = data[:4, 1:]
    def norm(x):
        return np.sqrt(np.dot(x,x))
    cosftn = lambda x, y : np.dot(x, y)/ norm(x)/norm(y)
    Euclidean = lambda x, y : np.sqrt(np.dot(x-y, x-y))
    graph = generate_graph(data, cosftn)
    
    #graph = get_example_graph()
    #params length = 2p 
    params = [1.0, 1.0]
    qaoa = Qaoa(graph)

    res = minimize(qaoa.execute_circ, 
                        params, 
                        method='COBYLA')
    print(res)