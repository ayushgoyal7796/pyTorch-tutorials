
# coding: utf-8

# In[1]:


# Data Parallel
# we will learn how to use multiple GPUs using DataParallel.
# It is very easy to use GPUs with pyTorch. we can put our model on a GPU:
    # device = torch.device("cuda:0")
    # model.to(device)


# In[2]:


# It is natural to execute forward, backward propagations on multiple GPUs.
# However, PyTorch will only use one GPU by default. 
# We can easily run our operations on multiple GPUs by making our model run parallelly using DataParallel
    # model = nn.DataParallel(model)


# In[3]:


# Imports and Parameters

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

input_size = 5
output_size = 2

batch_size = 30
data_size = 100


# In[4]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[5]:


# Dummy Dataset
class RandomDataset(Dataset):
    
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                        batch_size=batch_size, shuffle=True)


# In[6]:


# Simple Model
# Our model just gets an input, performs a linear operation, and gives an output.
# Print statement inside the model has been placed to monitor the size of input and output tensors.

class Model(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        
    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(), "output size", output.size())
        return output


# In[7]:


# create model and DataParallel
# Make a model instance and check if we have multiple GPUs.
# If we have multiple GPUs we can wrap our model using nn.DataParallel. Then we can put our model on GPUs by using model.to(device).

model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use ", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    
model.to(device)


# In[8]:


# Run the Model
# Now we can see the sizes of input and output tensors.
for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input_size", input.size(), "output_size", output.size())


# In[9]:


# Results
# If we have no GPU or one GPU, when we batch 30 inputs and 30 ouputs, the model gets 30 and outputs 30 as expected but if we have multiple GPUs we can get results like this:


# In[10]:


# 2 GPUs
#Let's use 2 GPUs!
#    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#    In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
#    In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
#Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])


# In[11]:


# 3 GPUs
#Let's use 3 GPUs!
#    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])


# In[12]:


# 8 GPUs
#Let's use 8 GPUs!
#    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])

