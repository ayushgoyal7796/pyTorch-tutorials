
# coding: utf-8

# In[1]:


# PyTorch is a Python based scientific computing package targetted at two sets of audiences : 
    # A replacement of NumPy to use the power of GPUs
    # A deep Learning research platform that provides maximum flexibility and speed


# In[2]:


# Tensors are similar to NumPy's ndarrays but can also be used on GPU to accelerate computing.


# In[3]:


from __future__ import print_function
import torch


# In[4]:


# Construct a 5 x 3 matrix, uninitialized
x = torch.empty(5, 3)
print(x)


# In[5]:


# Construct a randomly initialized matrix
x = torch.rand(5, 3)
print(x)


# In[6]:


# Construct a matrix filled with zeros and dtype long
x = torch.zeros(5, 3, dtype=torch.long)
print(x)


# In[7]:


# Construct a tensor directly from data
x = torch.tensor([5.5, 3])
print(x)


# In[8]:


# Create a tensor based on existing tensor. These methods will reuse the properties of input tensor, e.g. dtype, unless new values are provided.
x = x.new_ones(5, 3, dtype=torch.double)    # new_* method takes in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)   # override dtype!
print(x)                                     # result has the same size


# In[9]:


# Get its size
print(x.size())


# In[10]:


# Operations

# Addition

# syntax 1
y = torch.rand(5, 3)
print(x + y)

# syntax 2
print(torch.add(x, y))

# syntax 3 : providing an output tensor as argument
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# syntax 4 : addition in-place
# adds x to y
y.add_(x)
print(y)
# Any operation that mutates a tensor in-place is post-fixed with an _. For example: x.copy_(y), x.t_(), will change x.


# In[11]:


# you can use standard NumPy-like indexing
print(x)
print(x[:, 1])


# In[12]:


# Resizing / Reshaping tensor
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)   # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())


# In[13]:


# if you have one value tensor then use .item() to get the value as a Python number.
x = torch.randn(1)
print(x)
print(x.item())


# In[14]:


# Numpy Bridge
# Converting a Torch Tensor to a NumPy array and vice versa.
# The torch tensor and NumPy array will share their underlying memory locations, and changing one will change the other.

# Converting a Torch Tensor to a NumPy Array
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

# Converting NumPy Array to Torch Tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)


# In[15]:


# CUDA tensors
# tensors can be moved onto any device using .to method.

# this cell will run only if CUDA is available
# we will use "torch.device" objects to move tensors in and out of GPU

if torch.cuda.is_available():
    device = torch.device("cuda")    # CUDA device object
    y = torch.ones_like(x, device=device)    # directly create tensor on GPU
    x = x.to(device)    # or just use string '.to("cuda")'
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))    # '.to' can also change dtype together!
    

