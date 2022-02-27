## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I




#Suggestion from mentor:
#Max pooling is a simple down-sampling operation and I am sure you know what it does.
# nn.MaxPool2d returns a stateless (has no trainable parameters) object so you do not 
#have to declare multiple instances with the same kernel size, 
#you can just reuse the same object multiple times. :wink:
#The same goes for the dropout function (for a given dropping probability, of course).

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        #output size: (n+2p-f)/s+1
        #n:input_size,p:padding,f:kernel_size,s:stride
        
        #Layer 1: 
        #input: 1x224x224
        self.conv1 = nn.Conv2d(1, 32, 5)
        #output_conv1: (n+2p-f)/s+1 =(224+0*2-5)/1+1=220, (32,220,220)       
        
        #Layer 2:
        # maxpool layer with kernel_size=2, stride=2:
        # input_shape: (32,220,220) 
        self.pool1 = nn.MaxPool2d(2, 2)
        # (n+2p-f)/s+1 =(220+0*2-2)/2+1=110
        # output_shape: (32, 110,110)
        
        #Layer 3: 
        # drop layer with probalility 0.25
        self.drop1 = nn.Dropout(p=0.25)
        # input_shape,output_shape: (32, 110,110)
        
        # Layer 4:
        # input_shape: (32, 110,110)
        self.conv2 = nn.Conv2d(32, 64, 5)
        # (n+2p-f)/s+1 =(110+0*2-5)/1+1=106
        #output: (64,106,106)
        
        #Layer 5:
        # input_shape: (64, 106,106)
        self.pool2 = nn.MaxPool2d(2, 2)
        # (n+2p-f)/s+1 =(106+0*2-2)/2+1=53
        # output: (64,53,53)
        
        #Layer 6:
        # input_shape,output_shape: (64, 53,53)
        self.drop2 = nn.Dropout(p=0.25)
                
        # Layer 7:
        # input_shape: (64, 53,53)
        self.conv3 = nn.Conv2d(64, 32, 5)
        # (n+2p-f)/s+1 =(53+0*2-5)/1+1=49
        #output: (32,49,49)
        
        #Layer 8:
        #input_shape: (32,49,49)
        self.pool3 = nn.MaxPool2d(2, 2)
        # (n+2p-f)/s+1 =(49+0*2-2)/2+1=24
        # output: (32,24,24)
        
        
        #Layer 9:
        #input_shape,output_shape: (32,24,24)
        self.drop3 = nn.Dropout(p=0.25)
        
        # Layer 10:
        # input_shape: (32, 24,24)
        self.conv4 = nn.Conv2d(32, 32, 5)
        # (n+2p-f)/s+1 =(24+0*2-5)/1+1=20
        #output: (32,20,20)
        
        #Layer 11:
        #input_shape: (32,20,20)
        self.pool4 = nn.MaxPool2d(2, 2)
        # (n+2p-f)/s+1 =(12+0*2-2)/2+1=10
        # output: (32,10,10)
      
        #Layer 12:
        #input_shape,output_shape: (32,10,10)
        self.drop4 = nn.Dropout(p=0.2)
        
        # Layer 13:
        # input_shape: (32, 10,10)
        self.conv5 = nn.Conv2d(32, 32, 5)
        # (n+2p-f)/s+1 =(10+0*2-5)/1+1=6
        #output: (32,6,6)
        
        #Layer 14:
        #input_shape: (32,6,6)
        self.pool5 = nn.MaxPool2d(2, 2)
        # (n+2p-f)/s+1 =(6+0*2-2)/2+1=3
        # output: (32,3,3)
      
        #Layer 12:
        #input_shape,output_shape: (32,3,3)
        self.drop5 = nn.Dropout(p=0.2)
    
    # Layer 10: flat: x = x.view(x.size(0), -1)
        # Layer 11:
        #input:(12*26*26,) 
        self.fc1 = nn.Linear(32*3*3, 136)
        #self.fc2 = nn.Linear(320, 136)
        #output: (136, )
            
            
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # two conv/relu + pool layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.drop1(x)
        
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.drop2(x)
        
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.drop3(x)
        
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.drop4(x)
        
        x = self.pool5(F.relu(self.conv5(x)))
        x = self.drop5(x)

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        

        x = self.fc1(x)
        #x = self.fc2(x)
        # final output 

        # a modified x, having gone through all the layers of your model, should be returned
        return x