
# coding: utf-8

# In[1]:


# Autograd : Automatic Differentiation
# Autograd package provides automatic differentiation for all operations on tensors. It is a deine-by-run framework, which means that your backprop is defined by how your code is run and that every single iteration can be different.
# If we set torch.Tensor's attribute .requires_grad as True, it starts to track all operations on  it.
# When you finish your computation you can call .backward() and have all the gradients computed automatically. The gradients for this tensor will be accumulated into .grad attribute.
# To stop a tensor from tracking history, you can call .detach() to detach it from computation history, and to prevent future computation from being tracked.
# To prevent tracking history (and using memory), you can also wrap the code block in 'with torch.no_grad():'. This can be particularly helpful when evaluating a model because the model may have trainable parameters with requires_grad=True, but for which we don’t need the gradients.


# In[2]:


# Tensor and Function are interconnected and build up an acyclic graph, that encodes a complete history of computation. Each tensor has a .grad_fn attribute that references a Function that has created the Tensor except for Tensors created by the user - their grad_fn is None.


# In[3]:


# If you want to compute the derivatives, you can call .backward() on a Tensor. If Tensor is a scalar (i.e. it holds a one element data), you don’t need to specify any arguments to backward(), however if it has more elements, you need to specify a gradient argument that is a tensor of matching shape.


# In[4]:


import torch


# In[5]:


# create tensor and set requires_grad=True to track computation with it
x = torch.ones(2, 2, requires_grad=True)
print(x)


# In[6]:


# Do an operation on tensor
y = x + 2
print(y)


# In[7]:


# y was created as a result of an operation, so it has a grad_fn
print(y.grad_fn)


# In[8]:


# Do more operations on y
z = y*y*3
out = z.mean()
print(z)
print(out)


# In[9]:


# .requires_grad_() changes an existing Tensor's requires_grad flag in-place. The input flag defaults to False if not given.
a = torch.randn(2, 2)
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a*a).sum()
print(b.grad_fn)


# In[10]:


# Gradients
# lets backpropagate
# since out contains a single scalar, out.backward() is equivalent to out.backward(torch.tensor(1))
out.backward()


# In[11]:


# print gradient d(out)/dx
print(x.grad)


# In[12]:


x = torch.randn(3, requires_grad=True)
y = x*2
while y.data.norm() < 1000:
    y = y*2
print(y)


# In[13]:


gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(gradients)
print(x.grad)


# In[14]:


# you can also stop autorad from tracking history on tensors with .requires_grad=True by wraping the code block in with torch.no_grad():
print(x.requires_grad)
print((x**2).requires_grad)

with torch.no_grad():
    print((x**2).requires_grad)
