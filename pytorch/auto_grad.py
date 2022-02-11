import torch, torchvision

model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1,3,64,64) # input
labels =  torch.rand(1,1000) # a coloum vectoe labels required output
prediction = model(data) # forword pass
loss = (prediction - labels).sum() # calculte the loss
loss.backward() # backpropagate to compute the cost
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9) # optimize the model
optim.step() # gradient decent