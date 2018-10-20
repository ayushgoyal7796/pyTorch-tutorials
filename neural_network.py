
# coding: utf-8

# In[1]:


# Neural networks can be constructed using the torch.nn package
# nn depends on autograd to define models and differentiate them.
# nn.Module contains layers, and a method forward(input) that returns output.


# In[2]:


# Convnet
# It is a simple feed-forward network.It takes the input, feeds it through several layers one after the other, and then finally gives the output.


# In[3]:


# A typical training procedure for a neural network is as follows:
    # define the neural network that has some learnable parameters(or weights)
    # iterate over a dataset of inputs
    # process input through the network
    # compute the loss
    # propagate gradients back into the network's parameters
    # update the weights of the network, typically using a simple update rule weight = weight - learning_rate * gradient


# In[4]:


# Define the network:
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        # 1 input channel image, 6 output channels, 5 x 5 convolutions
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]   # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
net = Net()
print(net)


# In[5]:


# we just have to define the forward function, and the backward function (where gradients are computed) is automatically defined for us using autograd
# we can use any of the Tensor operations in the forward function.
# The learnable parameters of a model are returned by net.parameters()

params = list(net.parameters())
print(len(params))
print(params[0].size())    # conv1's .weights


# In[6]:


# Expected input size of this net is 32 x 32. To use this on MNIST dataset we need to resize the images to 32 x 32.
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)


# In[7]:


# zero the gradient buffers of all parameters and backprop with random gradients
net.zero_grad()
out.backward(torch.randn(1, 10))


# In[8]:


# torch.nn only supports mini-batches. The entire torch.nn package only supports inputs which are a mini-batch of samples and not a single sample.
# For example, nn.Conv2d will take in a 4D tensor of nSamples x nChannels x Height x Width.
# If we have a single sample, just use input.unsqueeze(0) to add a fake batch dimension.


# In[9]:


# Recap:
    # torch.Tensor: A multi-dimensional array with support for autograd operations like backward(). Also holds gradients w.r.t. the tensor.
    # nn.Module: Neural Network module. Convinient way of encapsulating parameters, with helpers for moving them to GPU, exporting, loading etc.
    # nn.Parameters: A kind of Tensor, that is automatically registered as a parameter when assigned as an attribute to Module.
    # autograd.Function: implements forward and backward definitions of an autograd operation. Every Tensor operation, creates at least a single Function node, that connects to functions that created a Tensor and encodes its history. 


# In[10]:


# we covered:
    # Defining a neural network.
    # Processing inputs and calling  backwards
# still left:
    # Computing the Loss
    # updating the weights of the network


# In[11]:


# Loss Function
# A loss function takes the (output, target) pair of inputs, and estimates how far away the output is from the target.
# There are several loss functions under the nn.package.
# A simple loss is nn.MSELoss which computes the mean-squared error between the input and the target.

print(input)
output = net(input)
print(output)
target = torch.randn(10)   # dummy target
print(target)
target = target.view(1, -1)   # make it the same shape as output
print(target)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)


# In[12]:


# now if we follow loss in backward direction, using its .grad_fn attribute, we will see a graph of computations that looks like this:
# input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
#       -> view -> linear -> relu -> linear -> relu -> linear
#       -> MSELoss
#       -> loss


# In[13]:


# so when we call loss.backward(), the whole graph is differentiated w.r.t. the loss, and all Tensors in the graph that has requires_grad=True will have their .grad Tensor accumulated with the gradient.

print(loss.grad_fn)   # MSELoss
print(loss.grad_fn.next_functions[0][0])   # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])    # ReLU


# In[14]:


# Backprop
# To backpropagate the error all we have to do is to loss.backward().
# You need to clear the existing gradients though, else gradients will be accumulated to existing gradients.
net.zero_grad()

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


# In[15]:


# Update the weights
# The simplest update rule used in practice is the Stochastic Gradient Descent (SGD):
    # weight = weight - learning_rate * gradient
# We can implement this using simple python code:
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data*learning_rate)


# In[16]:


# However, as we are using neural networks, we want to use various different update rules such as SGD, Nesterov-SGD, Adam, RMSProp etc.
# To enable this, a small package torch.optim implements all these methods.


# In[17]:


import torch.optim as optim

# create the optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in the training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()   # Does the update


# In[18]:


# Observe how gradient buffers had to be manually set to zero using optimizer.zero_grad().
# This is because gradients are accumulated as explained in Backprop section.

