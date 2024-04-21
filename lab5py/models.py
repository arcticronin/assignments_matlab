import torch.nn.functional as F
import torch.nn as nn

# from lesson
class Net_class(nn.Module):
    def __init__(self):
        super(Net_class, self).__init__() 
        self.conv1 = nn.Conv2d(3, 6, 5)  # First convolutional layer
        self.conv2 = nn.Conv2d(6, 16, 5)  # Second convolutional layer
        # pool
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) ## set nim ogf classes now

    def forward(self, x):
        """Define the forward pass of the network"""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.shape[0], -1)  # Flatten the output from convolutional layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

## performing 8 layer
class Net8(nn.Module):
    def __init__(self):
        super(Net8, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        self.dropoutl1 = nn.Dropout(0.25)
        self.dropoutl2 = nn.Dropout(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 *  2 * 2 , 128)  # 64 filters, 8x8 feature maps size after pooling
        self.fc2 = nn.Linear(128, 32)  # 10 classes for CIFAR-10
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        # Forward pass through convolutional layers with ReLU and pooling
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool(x)
        
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.pool(x)
        
        x = self.bn5(F.relu(self.conv5(x)))
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.pool(x)
        
        x = self.bn7(F.relu(self.conv7(x)))
        x = self.bn8(F.relu(self.conv8(x)))
        x = self.pool(x)
    
        # print(x.shape)
        x = x.view(-1, 256 * 2 * 2) 

        # Fully connected
        x = F.leaky_relu(self.fc1(x))
        x = self.dropoutl1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropoutl2(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1) 

## simple 4 

class Net_nbn(nn.Module):
    def __init__(self):
        super(Net_nbn, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 *  2 * 2 , 128) 
        self.fc2 = nn.Linear(128, 32) 
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        # print(x.shape)
        x = x.view(-1, 256 * 2 * 2) 
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class Net_rl(nn.Module):
    def __init__(self):
        super(Net_rl, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 *  2 * 2 , 128) 
        self.fc2 = nn.Linear(128, 32) 
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.pool(x)
        
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool(x)
        
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.pool(x)
        
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.pool(x)
        # print(x.shape)
        x = x.view(-1, 256 * 2 * 2) 
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class Net_nodrop(nn.Module):
    def __init__(self):
        super(Net_nodrop, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        #dropouts removed
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 *  2 * 2 , 128) 
        self.fc2 = nn.Linear(128, 32) 
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.pool(x)
        
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool(x)
        
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.pool(x)
        
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.pool(x)
        # print(x.shape)
        x = x.view(-1, 256 * 2 * 2) 
        
        # Fully connected
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)  


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        #dropouts
        self.dropoutl1 = nn.Dropout(0.25)
        self.dropoutl2 = nn.Dropout(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 *  2 * 2 , 128) 
        self.fc2 = nn.Linear(128, 32) 
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.pool(x)
        
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool(x)
        
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.pool(x)
        
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.pool(x)
        # print(x.shape)
        x = x.view(-1, 256 * 2 * 2) 
        
        # Fully connected
        x = F.leaky_relu(self.fc1(x))
        x = self.dropoutl1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropoutl2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)  
        
# big linear computation (github)
class Net_linear(nn.Module):
    def __init__(self):
        super(Net_linear, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(3,3), padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(3,3), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(3,3), padding=(1,1))
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=(3,3), padding=(1,1))
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(in_features=8*8*256, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.Dropout = nn.Dropout(0.25)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) #32*32*48
        x = F.relu(self.conv2(x)) #32*32*96
        x = self.pool(x) #16*16*96
        x = self.Dropout(x)
        x = F.relu(self.conv3(x)) #16*16*192
        x = F.relu(self.conv4(x)) #16*16*256
        x = self.pool(x) # 8*8*256
        x = self.Dropout(x)
        x = x.view(-1, 8*8*256) # reshape x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.Dropout(x)
        x = self.fc3(x)
        return x


# cererus
class Cerberus(nn.Module):
    def __init__(self):
        super(Cerberus, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(256)
        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(256)
        self.conv11 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(256)
        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(256)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout layers
        self.dropoutl = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 *  4 * 4 , 128)  # 64 filters, 8x8 feature maps size after pooling
        self.fc2 = nn.Linear(128, 10)  # 10 classes for CIFAR-10

    def forward(self, x):
        # Forward pass through convolutional layers with ReLU and pooling
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.pool(x)
        # x = self.dropout2(x)

        x = self.bn5(F.relu(self.conv5(x)))
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.bn7(F.relu(self.conv7(x)))
        x = self.bn8(F.relu(self.conv8(x)))
        x = self.pool(x)
        # x = self.dropout4(x)

        x = self.bn9(F.relu(self.conv9(x)))
        x = self.bn10(F.relu(self.conv10(x)))
        x = self.bn11(F.relu(self.conv11(x)))
        x = self.bn12(F.relu(self.conv12(x)))
        x = self.pool(x)


        # Flattening the layer
        #print(x.shape)
        x = x.view(-1, 256 * 4 * 4)  # Flatten the tensor for the fully connected layer

        # Fully connected layer with ReLU and dropout
        x = F.leaky_relu(self.fc1(x))
        x = self.dropoutl(x)
        x = self.fc2(x)  # Output layer, no activation function here
        
        return F.log_softmax(x, dim=1)  # Applying softmax to get probabilities


## improved cerberus
class Cerberu2(nn.Module):
    def __init__(self):
        super(Cerberu2, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(256)
        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(256)
        self.conv11 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(256)
        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(256)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout layers
        self.dropoutl1 = nn.Dropout(0.25)
        self.dropoutl2 = nn.Dropout(0.5)
        
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 *  4 * 4 , 512)  # 64 filters, 8x8 feature maps size after pooling        
        self.fc2 = nn.Linear(512 , 64)  # 64 filters, 8x8 feature maps size after pooling
        self.fc3 = nn.Linear(64, 10)  # 10 classes for CIFAR-10

    def forward(self, x):
        # Forward pass through convolutional layers with ReLU and pooling
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.pool(x)
        
        x = self.bn5(F.relu(self.conv5(x)))
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.bn7(F.relu(self.conv7(x)))
        x = self.bn8(F.relu(self.conv8(x)))
        x = self.pool(x)
        
        x = self.bn9(F.relu(self.conv9(x)))
        x = self.bn10(F.relu(self.conv10(x)))
        x = self.bn11(F.relu(self.conv11(x)))
        x = self.bn12(F.relu(self.conv12(x)))
        x = self.pool(x)


        # Flattening the layer
        #print(x.shape)
        x = x.view(-1, 256 * 4 * 4)  # Flatten the tensor for the fully connected layer

        # Fully connected layer with ReLU and dropout
        x = F.leaky_relu(self.fc1(x))
        x = self.dropoutl1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropoutl2(x)
        x = self.fc3(x)  # Output layer, no activation function here
        
        return F.log_softmax(x, dim=1)  # Applying softmax to get probabilities