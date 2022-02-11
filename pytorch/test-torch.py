import torch as tr
import numpy as np



input = [1,2,3]
weight = [1,2,3]
input_tensor = tr.tensor(input)
wights_tensor = tr.tensor(weight)

# w^.x matrix multiplicarion

z =  input_tensor @ wights_tensor.T 
print(z)