import numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter
import pickle as pkl
import random
import pdb
import nltk
from nltk.util import ngrams
from torch.utils.data.sampler import SubsetRandomSampler
random.seed(15)

import string

PAD_IDX = 52948
UNK_IDX = 77808
BATCH_SIZE = 32

def test_model(model, train=False):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total_sample = 0
    total_loss = 0
    
    model.eval()
    # get a random sample
    if train:
        loader = torch.utils.data.DataLoader(
            SNLIDataset(train_sent1_indices, train_sent2_indices, train_label),batch_size=BATCH_SIZE,collate_fn=SNLIvocab_collate_func,
            sampler=SubsetRandomSampler(range(10*BATCH_SIZE)))
    else:
        loader = torch.utils.data.DataLoader(
            SNLIDataset(val_sent1_indices, val_sent2_indices, val_label),batch_size=BATCH_SIZE,collate_fn=SNLIvocab_collate_func,
            sampler=SubsetRandomSampler(range(10*BATCH_SIZE)))

    for i, sample in enumerate(loader):
            size = sample[0].shape[0]
            #print(sample[0].shape)
            outputs = F.softmax(model(sample[0], sample[1]), dim=1)
            minibatch_loss = criterion(outputs, sample[2])
            total_loss += minibatch_loss.item()
            predicted = outputs.max(1, keepdim=True)[1].view(-1)
            
            total_sample += size
            label = sample[2]
            correct += predicted.eq(label.view_as(predicted)).sum().item()

    total_batch = i + 1
    acc = 100 * correct / total_sample
    los = total_loss / total_batch
    return acc, los

##load back all necessary data:
all_words = pkl.load(open('../all_words.p','rb'))
word2idx = pkl.load(open('../word2idx.p','rb'))
idx2word = pkl.load(open('../idx2word.p','rb'))
word2vec = pkl.load(open('../word2vec.p','rb'))
weights_matrix = pkl.load(open('../weights_matrix.p','rb'))

train_sent1_indices = pkl.load(open("../train_sent1_indices.p", "rb"))
train_sent2_indices = pkl.load(open("../train_sent2_indices.p", "rb"))
val_sent2_indices = pkl.load(open("../val_sent2_indices.p", "rb"))
val_sent1_indices = pkl.load(open("../val_sent1_indices.p", "rb"))

train_label = pkl.load(open("../train_label.p", "rb"))
val_label = pkl.load(open("../val_label.p", "rb"))

MAX_SENTENCE_LENGTH = 25
class SNLIDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, sent1_data, sent2_data, target_list):
        """
        @param data_list: list of character
        @param target_list: list of targets

        """
        self.sent1_data = sent1_data
        self.sent2_data = sent2_data
        self.target_list = target_list
        assert (len(self.sent1_data) == len(self.target_list))
        assert (len(self.sent2_data) == len(self.target_list))

    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        item = dict()
        
        sent1_index_list = self.sent1_data[key][:MAX_SENTENCE_LENGTH]
        sent2_index_list = self.sent2_data[key][:MAX_SENTENCE_LENGTH]
        label = self.target_list[key]
        return [sent1_index_list, sent2_index_list, label]
    
#note since PAD is already in dataset, here we need to pad with PAD_IDX not 0
def SNLIvocab_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all
    data have the same length
    """
    sent1_list = []
    sent2_list = []
    label_list = []

    for datum in batch:
        label_list.append(datum[2])
        
    # padding
    for datum in batch:
        padded_vec0 = np.pad(np.array(datum[0]),
                                pad_width=((0 ,MAX_SENTENCE_LENGTH-len(datum[0]))),
                                mode="constant", constant_values = PAD_IDX)
        sent1_list.append(list(padded_vec0))
    
        padded_vec1 = np.pad(np.array(datum[1]),
                                pad_width=((0 ,MAX_SENTENCE_LENGTH-len(datum[1]))),
                                mode="constant", constant_values= PAD_IDX)
        sent2_list.append(list(padded_vec1))

    return [torch.from_numpy(np.array(sent1_list)),torch.from_numpy(np.array(sent2_list)), torch.LongTensor(label_list)]


BATCH_SIZE = 32
train_dataset = SNLIDataset(train_sent1_indices, train_sent2_indices, train_label)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                          batch_size = BATCH_SIZE,
                                          collate_fn = SNLIvocab_collate_func,
                                          shuffle = True)

val_dataset = SNLIDataset(val_sent1_indices, val_sent2_indices, val_label)
val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                        batch_size = BATCH_SIZE,
                                        collate_fn = SNLIvocab_collate_func,
                                        shuffle = False)

class RNN(nn.Module):
    def __init__(self, weights_matrix, hidden_size, num_layers, num_classes):
        # RNN Accepts the following hyperparams:
        # emb_size: Embedding Size
        # hidden_size: Hidden Size of layer in RNN
        # num_layers: number of layers in RNN
        # num_classes: number of output classes
        # vocab_size: vocabulary size
        super().__init__()

        self.num_layers, self.hidden_size = num_layers, hidden_size
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix),freeze=True)
        emb_size = weights_matrix.shape[1]
        
        self.rnn1 = nn.GRU(emb_size, hidden_size, num_layers, batch_first=True, dropout=0.5, bidirectional=True)
        self.rnn2 = nn.GRU(emb_size, hidden_size, num_layers, batch_first=True, dropout=0.5, bidirectional=True)
        
        self.fc1 = nn.Linear(hidden_size * 2 * 2, 300)
        self.dropout1 = nn.Dropout(0.3) 
        self.out = nn.Linear(300, num_classes)
        
    
        
    def init_hidden(self, batch_size):
        # Function initializes the activation of recurrent neural net at timestep 0
        # Needs to be in format (num_layers, batch_size, hidden_size)
        hidden = torch.randn(self.num_layers*2, batch_size, self.hidden_size)

        return hidden

    def forward(self, sent1, sent2):
        # reset hidden state

        batch_size = BATCH_SIZE

        self.hidden1 = self.init_hidden(batch_size)
        self.hidden2 = self.init_hidden(batch_size)

        # get embedding of characters
        sent1_embed = self.embedding(sent1)
        sent2_embed = self.embedding(sent2)
        
        # fprop though RNN
        sent1_rnn_out, self.hidden1 = self.rnn1(sent1_embed, self.hidden1)
        sent2_rnn_out, self.hidden2 = self.rnn2(sent2_embed, self.hidden2)
       
    
        # sum hidden activations of RNN across time
        sent1_rnn_out = torch.sum(sent1_rnn_out, dim=1)
        sent2_rnn_out = torch.sum(sent2_rnn_out, dim=1)
        
        rnn_out = torch.cat([sent1_rnn_out, sent2_rnn_out], 1)
        #print(rnn_out.shape)
        
        x = self.fc1(rnn_out)
        x = F.relu(x)
        x = self.dropout1(x)
        
        logits = self.out(x)
        return logits


#create model:
model = RNN(weights_matrix=weights_matrix, hidden_size=300, num_layers=1, num_classes=3)

learning_rate = 3e-4
num_epochs = 12 # number epoch to train

# Criterion and Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



#training and save stats and best model
RNNval_accuracy = []
RNNval_loss = []
RNNtrain_accuracy = []
RNNtrain_loss = []
best_val_acc = 0
for epoch in range(num_epochs):
    for i, sample in enumerate(train_loader):
#         print(len(sample['label']))
        model.train()
        optimizer.zero_grad()
        # Forward pass
        output = model(sample[0], sample[1])
        #print(output)
        label = sample[2]
        loss = criterion(output, label)

        # Backward and optimize
        loss.backward()
        optimizer.step()
        # validate every 100 iterations
        if i > 0 and i % 300 == 0:
            #validate
            val_acc, val_los = test_model(model, train=False)
            tra_acc, tra_los = test_model(model, train=True)
            RNNval_accuracy.append(val_acc)
            RNNval_loss.append(val_los)
            RNNtrain_accuracy.append(tra_acc)
            RNNtrain_loss.append(tra_los)
            print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}'.format(
                       epoch+1, num_epochs, i+1, len(train_loader), val_acc))
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print('find new record, save the model!')
                torch.save(model.state_dict(), 'RNN300_best_model.pkl')
            
OUT_DICT = {'val_accuracy':RNNval_accuracy, 'val_loss': RNNval_loss, 'train_accuracy':RNNtrain_accuracy, 'train_loss':RNNtrain_loss, 'trainable_parameters': sum([np.prod(p.size()) for p in model.parameters()if p.requires_grad ])}

pkl.dump(OUT_DICT, open("SNL_RNN_HIDDEN300.p", "wb"))
  
