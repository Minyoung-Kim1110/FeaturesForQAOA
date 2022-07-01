''' 
Author: Minyoung Kim ( Jul. 1, 2022)
2022 Hackaton 
Team: ThankQ
description: 
    functions 
''' 

import torch 

def bitstring_to_int(bit_string_sample):
    bit_string = "".join(str(bs) for bs in bit_string_sample)
    return int(bit_string, base=2)
# Define distance function 

def sin_square(x:torch.Tensor, y:torch.Tensor):
    return 1- torch.dot(x, y)**2/torch.dot(x, x)/torch.dot(y,y)
