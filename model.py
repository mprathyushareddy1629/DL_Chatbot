import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        #it ia a feed forward neural network with 2 layers and softmax for getting propbablities for each classes


        super(NeuralNetwork,self).__init__()
        self.l1=nn.Linear(input_size,hidden_size)
        self.l2=nn.Linear(hidden_size,hidden_size) 
        self.l3=nn.Linear(hidden_size,num_classes)  
        self.relu=nn.ReLU()

    def forward(self,x):
        out=self.l1(x)
        out=self.relu(out)
        out=self.l2(out)
        out=self.relu(out)
        out=self.l3(out)
        out = nn.functional.softmax(out, dim=1)
        #no activation and no softmax

        return out
