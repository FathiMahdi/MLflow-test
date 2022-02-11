from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        #initialize the model parameters
        # input channel (gray), output channel ,kernel size
        self.conv1 = nn.Conv2d(1,6,5) #define the first conv layer
        self.conv2 = nn.Conv2d(6,16,5) #define the first conv layer
        # y = Wx+b # fully connected layers
        self.fc1 = nn.Linear(16*5*5,120) # 16 layers and 5*5 image
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10) # last prediction layer (no conv)

    def forward(self,x):
        # max pooling over a (2,2) window
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv1(x)),(2)) # for square iput
        x = torch.flatten(x,1) # flatten all the dimentions except the batch dimention
        x = F.relu(self.fc1(x)) # pass x to the first fully connected layer
        x = F.relu(self.fc2(x)) # pass x to the first fully connected layer
        x = self.fc3(x) # pridiction layer
        return x

net = Net()
params = list(net.parameters())
print(params[0].size())  # conv1's .weight
data = np.full((32, 32, 1), 255, dtype = np.uint8)
data = torch.from_numpy(data)
#data = torch.randn(1,1,32,32)
out = net(data)
#print(data.shape)

# zero the grad 

net.zero_grad()

# backpropagate with random gradient

# calculate the loss

criterion = nn.MSEloss()
loss  = criterion(output, targer)

######
#optimization

optimizer = optim.SGD(net.parameters(),lr=0.01)
optimizer.zero_grad()
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update