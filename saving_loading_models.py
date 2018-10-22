
# coding: utf-8

# In[1]:


# Saving and Loading models
# There are three core functions for saving and loading models:
    # torch.save: saves a serialized object to disk. It uses pickle utility for serialization. Models, tensors, and dictionaries of all kinds of objects can be saved using this function.
    # torch.load: uses pickle's unpickling facilities to deserialize pickled object files to memory. This function also facilitates the device to load the data into.
    # torch.nn.Module.load_state_dict: Loads a model's parameter dictionary using a deserialized state_dict.


# In[2]:


# What is a state_dict ?
# Learnable parameters (weights and biases) of a torch.nn.Module model is contained in the model's parameters(accessed with model.parameters()).
# A state_dict is simply a python dictionary object that maps each layer to its parameter tensor.
# Only layers with learnable parameters (convolutional layers, linear layers, etc.) have entries in the model's state dict.
# Optimizer objects (torch.optim) also have a state_dict, which contains information about optimizer's state, as well as the hyperparameters used.


# In[3]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define model
class TheModelClass(nn.Module):
    
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# Initialize the model
model = TheModelClass()

# Initialize the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Print model's state_dict
print("Model's state dict:")
for param_tensor in model.state_dict():
    print(param_tensor, '\t', model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state dict:")
for var_name in optimizer.state_dict():
    print(var_name, '\t', optimizer.state_dict()[var_name])


# In[4]:


# saving and loading models for inference
# save/load state_dict (Recommended)

# save:
    # torch.save(model.state_dict, PATH)

# load:
    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))
    # model.eval()

# when saving a model for inferernce, it is only necessary to save the trained model’s learned parameters.
# Saving the model’s state_dict with the torch.save() function will give you the most flexibility for restoring the model later, which is why it is the recommended method for saving models.
# A common PyTorch convention is to save models using either a .pt or .pth file extension.

# remember we must call model.eval() to set dropout and normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results.


# In[5]:


# Note:
    # the load_state_dict() function takes a dictionary object, not a path to a saved object.
    # this means that you must deserialize the saved state_dict before you pass it to the load_state_dict() function.
    # we cannot use model.load_state_dict(PATH)


# In[6]:


# save/load entire model

# save:
    # torch.save(model, PATH)
    
# load:
    # Model class must be defined somewhere
    # model = torch.load(PATH)
    # model.eval()
    
# saving this way will save the entire module using Python's pickle module.
# disadvantage: the serialized data is bound to the specific classes and the exact directory structure used when the model is saved.
# The reason for this is because pickle does not save the model class itself.
# Rather, it saves a path to the file containing the class, which is used during load time.
# Because of this, your code can break in various ways when used in other projects or after refactors.

# A common PyTorch convention is to save models using either a .pt or .pth file extension.
# Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference.
# Failing to do this will yield inconsistent inference results.


# In[7]:


# saving and loading a general checkpoint for inference and/or resuming training

# save:
    # torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    #     'loss': loss,
    #     ...
    # }, PATH)
    
# load:
    # model = TheModelClass(*args, **kwargs)
    # optimizer = TheOptimizerClass(*args, **kwargs)
    #
    # checkpoint = torch.load(PATH)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])    
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
    #
    # model.eval()
    #   - or -
    # model.train()
    
# when saving a general checkpoint, to be used for either inference or resuming training you must save more than just the model's state_dict.
# it is also important to save the optimizer's state_dict, as this contains buffers and parameters that is updated as the model trains.
# we may also want to save the epoch we  left off on, the latest recorded training loss etc.

# To save multiple components, organize them in a dictionary and use torch.save() to serialize the dictionary.
# A common PyTorch convention is to save these checkpoints using the .tar file extension.
# To load, first initialize the model and optimizer, then load the dictionary locally using torch.load().

# Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference.
# Failing to do this will yield inconsistent inference results.
# If you wish to resuming training, call model.train() to ensure these layers are in training mode.


# In[8]:


# saving multiple models in one file
# save:
    # torch.save({
    #             'modelA_state_dict': modelA.state_dict(),
    #             'modelB_state_dict': modelB.state_dict(),
    #             'optimizerA_state_dict': optimizerA.state_dict(),
    #             'optimizerB_state_dict': optimizerB.state_dict(),
    #             ...
    #             }, PATH)
    
# load:
    # modelA = TheModelAClass(*args, **kwargs)
    # modelB = TheModelBClass(*args, **kwargs)
    # optimizerA = TheOptimizerAClass(*args, **kwargs)
    # optimizerB = TheOptimizerBClass(*args, **kwargs)
    # 
    # checkpoint = torch.load(PATH)
    # modelA.load_state_dict(checkpoint['modelA_state_dict'])
    # modelB.load_state_dict(checkpoint['modelB_state_dict'])
    # optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
    # optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])
    # 
    # modelA.eval()
    # modelB.eval()
    # # - or -
    # modelA.train()
    # modelB.train()
    
# When saving a model comprised of multiple torch.nn.Modules, such as a GAN, a sequence-to-sequence model, or an ensemble of models, you follow the same approach as when you are saving a general checkpoint.
# Save a dictionary of each model’s state_dict and corresponding optimizer. 
# A common PyTorch convention is to save these checkpoints using the .tar file extension.
# To load the models, first initialize the models and optimizers, then load the dictionary locally using torch.load().
# we can easily access the saved items by simply querrying the dictionary.
# we must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference.
# Failing to do this will yield inconsistent inference results.
# if you want to resume training call model.train() to set layers to training mode.


# In[9]:


# Warmstarting model using parameters from a different model
# Save:
    # torch.save(modelA.state_dict, PATH)
# Load:
    # modelB = TheModelClass(*args, **kwars)
    # modelB.load_state_dict(torch.load(PATH), strict=False)

# Partially loading a model or loading a partial model are common scenarios when transfer learning or training a new complex model.
# Leveraging trained parameters, even if only a few are usable, will help to warmstart the training process and hopefully help our model converge much faster than training from scratch.
# Whether you are loading from a partial state_dict, which is missing some keys, or loading a state_dict with more keys than the model that you are loading into, you can set the strict argument to False in the load_state_dict() function to ignore non-matching keys.

# If you want to load parameters from one layer to another, but some keys do not match, simply change the name of the parameter keys in the state_dict that you are loading to match the keys in the model that you are loading into.


# In[10]:


# Saving and Loading models across devices

# save on GPU, load on CPU
# save:
    # torch.save(model.state_dict(), PATH)
# load:
    # device = torch.device('cpu')
    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH, map_location=device))
# When loading a model on a CPU that was trained with a GPU, pass torch.device('cpu') to the map_location argument in the torch.load() function.

# save on GPU, load on GPU
# save:
    # torch.save(model.state_dict(), PATH)
# load:
    # device = torch.device("cuda")
    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))
    # model.to(device)
# Make sure to call input = input.to(device) on any input tensors that you feed to the model

# save on CPU, load on GPU
# save:
    # torch.save(model.state_dict(), PATH)
# load:
    # device = torch.device("cuda")
    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
    # model.to(device)   # to convert the model’s parameter tensors to CUDA tensors. 
# Make sure to call input = input.to(device) on any input tensors that you feed to the model

# saving torch.nn.DataParallel models
# save: 
    # torch.save(model.module.state_dict(), PATH)
# load:
    # load to whatever device you want
# torch.nn.DataParallel is a model wrapper that enables parallel GPU utilization.
# To save a DataParallel model generically, save the model.module.state_dict().
# This way, you have the flexibility to load the model any way you want to any device you want.

