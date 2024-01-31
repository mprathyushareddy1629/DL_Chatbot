import json 

from nltk_utils import tokenize,stem,bag_of_words
import numpy as np
import multiprocessing as mp


import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

from model import NeuralNetwork


with open('intents.json','r',encoding='utf-8') as f:
    intents=json.load(f)


all_words=[] #to store bag of words
tags = []
xy = []

for intent in intents['intents']:
    tag= intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))

ignore_words=['?','!','.',',']
all_words=[stem(w) for w in all_words if w not in ignore_words]
all_words=sorted(set(all_words))
tags=sorted(set(tags))
# print(tags)
# print(all_words)
X_train =[]
y_train=[]
for (pattern_sentence,tag) in xy:
    bag=bag_of_words(pattern_sentence,all_words)
    X_train.append(bag)

    label= tags.index(tag)
    y_train.append(label)#you will need one-hot encoding vector , but in this case we are using pytorch and later we use cross-entrophy loss, so we oly want class label


X_train=np.array(X_train)
y_train=np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples=len(X_train)
        self.x_data=X_train
        self.y_data=y_train

    
    #dataset[idx]
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    

#hyperparameters
batch_size=6
hidden_size=6
output_size=len(tags)
input_size=len(X_train[0])
learning_rate=0.001
num_epochs=1000
# print(input_size,len(all_words))
# print(output_size,tags)
if __name__ == '__main__':
    mp.set_start_method('spawn')  
    dataset = ChatDataset()  # Create an instance of the dataset
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
#for multithreading


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNetwork(input_size, hidden_size,output_size).to(device)


    #loss and optimizer

    criterion = nn.CrossEntropyLoss()
    optimizer= torch.optim.Adam(model.parameters(),lr=learning_rate )

    for epoch in range(num_epochs):
        total_correct = 0
        total_samples = 0
        for (words,labels) in train_loader:
            words = words.to(device)
            labels = labels.long().to(device)


            #forward

            outputs= model(words)
            loss = criterion(outputs,labels)

            #backward and optimizer step

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

        accuracy = total_correct / total_samples


        if (epoch +1) % 100==0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

    print(f'Final Loss: {loss.item():.4f}, Final Accuracy: {accuracy:.4f}')


    data= {
        "model_state":model.state_dict(),
        "input_size":input_size,
        "output_size":output_size,
        "hidden_size":hidden_size,
        "all_words":all_words,
        "tags":tags
    }

    FILE="data.pth"
    torch.save(data,FILE)

    print(f'training complete . file saved to {FILE}')

