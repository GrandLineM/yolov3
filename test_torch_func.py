import torch.nn as nn
import torch

x = torch.tensor([1., 2.], requires_grad=True)
y = 100*x

loss = y.sum()

print ("y",y)
print ("loss: ",loss)

print(x.grad)
loss.backward()
print(x.grad)

