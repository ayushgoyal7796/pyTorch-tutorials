
# coding: utf-8

# In[1]:


# Training a Classifier
# DATA
# Generally, when you have to deal with image, text, audio or video data, you can use standard python packages that load data into a numpy array. Then you can convert this array into a torch.Tensor.
    # For images, packages such as Pillow, OpenCV are useful
    # For audio, packages such as scipy and librosa
    # For text, either raw Python or Cython based loading, or NLTK and SpaCy are useful


# In[2]:


# Specifically for vision, a package called torchvision has data loaders for common datasets such as Imagenet, CIFAR10, MNIST etc. and data transformation for images, viz., torchvision.datasets and torch.utils.data.DataLoader.


# In[3]:


# Training an image Classifier
# we will do the following steps in order:
    # Load and normalizing the CIFAR10 training and test datasets using torchvision
    # Define a CNN
    # Define a loss function
    # Train the network on the training data
    # Test the network on the test data


# In[4]:


# Loading and normalizing CIFAR10

import torch
import torchvision
import torchvision.transforms as transforms


# In[5]:


# The output of torchvision datasets are PILImage images of range [0, 1]. We transform them to Tensors of normalized range [-1, 1].

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                        shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[6]:


# Showing some of the training images

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img/2 + 0.5    # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.pause(0.01)

# get some random training images

dataiter = iter(trainloader)
print(dataiter.next(), '\n\n')

images, labels = dataiter.next()

print(torchvision.utils.make_grid(images).shape, '\n\n')    # 3 x (2 + 32 + 2) x (2 + 32 + 2 + 32 + 2 + 32 +2 + 32 + 2)

# show images
imshow(torchvision.utils.make_grid(images))

# print labels
print(' '.join('%10s' % classes[labels[j]] for j in range(4)))


# In[7]:


# Define a CNN

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
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

net = Net()


# In[8]:


# Define a loss function and optimizer
# using cross-Entropy Loss and SGD with momentum

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# In[9]:


print(len(trainset))
print(len(trainloader))   # batch_size = 4


# In[10]:


# Train the network

for epoch in range(10):    # loop over the dataset multiple times
    
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get in inputs
        inputs, labels = data
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i%2000==1999:    # print every 2000 mini_batches
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/2000))
            running_loss = 0.0
print('Finished Training')


# In[11]:


# lets display an image from the testset
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)) )


# In[12]:


# Test the network on the test data

outputs = net(images)

# The outputs are the energies for the 10 classes.
# Higher the energy for a class, the more the network thinks that the image is of that particular class.
# we are going to get the index of the highest energy class:
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%10s' % classes[predicted[j]] for j in range(4)))


# In[13]:


# Lets see how the network performs on the whole testset

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
        
print('Accuracy of the network on the 10000 test images: %d %%' % (100*correct/total))


# In[14]:


# Class wise acuracy:

class_correct = list(0 for i in range(10))
print(class_correct)
class_total = list(0 for i in range(10))
print(class_total)

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels)
        for i in range(4):
            label = labels[i]
            class_correct[label]+=c[i].item()
            class_total[label]+=1

print(class_correct)
print(class_total)
for i in range(10):
    print('Accuracy of %5s : %2d %%'  % (classes[i], 100*class_correct[i]/class_total[i]))


# In[15]:


# Training on GPU
# Just like how we transfer a Tensor on to the GPU, we transfer the neural net onto the GPU.
# These methods will recursively go over all modules and convert their parameters and buffers to CUDA tensors:

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net.to(device)

# Remember that you will have to send the inputs and targets at every step to the GPU too:

    # inputs, labels = inputs.to(device), labels.to(device)

