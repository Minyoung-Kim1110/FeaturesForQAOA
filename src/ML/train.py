"""Use pennylane and pytorch library 
train network such that it maximizes the performance of QAOA

"""
#from locale import nl_langinfo
import pennylane as qml
import torch.nn as nn
from torch.optim import Adam
import torch
from dense_net import DenseNet
import numpy as np


# unitary operator U_B with parameter beta
def U_B(beta, n_wires):
    for wire in range(n_wires):
        qml.RX(2 * beta, wires=wire)

# unitary operator U_C with parameter gamma
def U_C(gamma, graph):
    for edge in graph:
        wire1 = edge[0]
        wire2 = edge[1]
        qml.CNOT(wires=[wire1, wire2])
        qml.RZ(gamma, wires=wire2)
        qml.CNOT(wires=[wire1, wire2])

def bitstring_to_int(bit_string_sample):
    bit_string = "".join(str(bs) for bs in bit_string_sample)
    return int(bit_string, base=2)


pauli_z = torch.tensor([[1., 0.], [0., -1.]], requires_grad=False).to(torch.float64)
pauli_z_2 = torch.kron(pauli_z, pauli_z)


def qaoa_circuit(graph, n_wires, edge=None, n_layers=1):
    # apply Hadamards to get the n qubit |+> state
    for wire in range(n_wires):
        qml.Hadamard(wires=wire)
    # p instances of unitary operators
    for i in range(n_layers):
        U_C(net.gammas[i], graph)
        U_B(net.betas[i], n_wires)
    if edge is None:
        # measurement phase
        return qml.sample()
    # during the optimization phase we are evaluating a term
    # in the objective using expval
    return qml.expval(qml.Hermitian(pauli_z_2, wires = edge))


def qaoa_maxcut(graph, n_wires: int, net: DenseNet, n_layers=1, print_log = True):
    dev = qml.device("default.qubit", wires=n_wires, shots=128)

    # initialize the parameters near zero
    print(f"initial params = {net.betas}, {net.gammas}")

    # Define distance function 
    def sin_square(x, y):
        return 1- np.dot(x, y)**2/np.dot(x, x)/np.dot(y,y)
    # minimize the negative of the objective function
    def objective(net: DenseNet):
        neg_obj = 0
        dists = np.zeros((n_wires,n_wires))
        zzs = np.zeros_like(dists)
        for i in range(n_wires):
            for j in range(i+1, n_wires):
                u, v = net(i), net(j)
                uv = (u * v).sum()
                uu = (u * u).sum()
                vv = (v * v).sum()
                dist = 1 - uv * uv / (uu * vv)
                dists[i][j], dists[j][i] = dist.item(), dist.item()
                circuit = qml.QNode(qaoa_circuit, dev, interface = "torch")
                zz = circuit(edge = (i, j), graph = graph, n_wires = n_wires, n_layers=n_layers)
                zzs[i][j], zzs[j][i] = zz.item(), zz.item()
                if ((i, j) in graph) or ((j, i) in graph):
                    neg_obj -= dist * (1-zz) / 2
                else:
                    neg_obj -= - dist*zz
        if print_log:
            print("embed = ")
            for i in range(3):
                print(net(i))
            print("dists = ")
            for a in dists:
                print(a)
            print("neg_obj = ", neg_obj)
            print("zzs = ", zzs)
            print(net.betas)
            print(net.gammas)
        return neg_obj

    # initialize optimizer: Adagrad works well empirically
    opt = Adam(net.parameters(), lr=0.01)
    net.train()

    # optimize parameters in objective
    steps = 1000
    for i in range(steps):
        opt.zero_grad()
        loss = objective(net)
        loss.backward()
        opt.step()
        if (i + 1) % 10 == 0:
            print("Objective after step {:5d}: {: .7f}".format(i + 1, -loss))
            
    return -objective(net)


# perform qaoa on our graph with p=1,2 and
# keep the bitstring sample lists

n_wires = 4
n_layers = 1

arr = np.load("../data/drug145_5.npy")[:n_wires, :]
label = arr[:, 0]
data = arr[:, 1:]

# generate bipartite graph from labels
graph = []
for i in range(n_wires):
    for j in range(i+1, n_wires):
        if label[i] != label[j]:
            graph.append((i, j))
print(f"graph = {graph}")
net = DenseNet(data, n_wires, n_layers = 3)
max_cut = qaoa_maxcut(graph, n_wires, net, n_layers=3, print_log=True)
print(f"print final result {max_cut}")
# results

# embed = 
# tensor([-0.0215, -0.1740, -0.0336], grad_fn=<AddBackward0>)
# tensor([ 0.0998, -0.0352,  0.0730], grad_fn=<AddBackward0>)
# tensor([-0.0215, -0.1740, -0.0336], grad_fn=<AddBackward0>)
# dists = 
# [0, 0.9955480694770813, 0.0, 0.9998735189437866]
# [0.9955480694770813, 0, 0.9955480694770813, 0.006074845790863037]
# [0.0, 0.9955480694770813, 0, 0.9998735189437866]
# [0.9998735189437866, 0.006074845790863037, 0.9998735189437866, 0]
# neg_obj =  tensor(-3.9584, dtype=torch.float64, grad_fn=<SubBackward0>)
# zzs =  [[0, -1.0, 1.0, -0.96875], [-1.0, 0, -0.984375, 0.96875], [1.0, -0.984375, 0, -1.0], [-0.96875, 0.96875, -1.0, 0]]
# Parameter containing:
# tensor([0.5363, 0.1956, 0.2708], dtype=torch.float64, requires_grad=True)
# Parameter containing:
# tensor([-0.7900, -0.7431, -0.5824], dtype=torch.float64, requires_grad=True)

