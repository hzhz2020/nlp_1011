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
                                mode="constant", constant_values= PAD_IDX)
        sent1_list.append(list(padded_vec0))
    
        padded_vec1 = np.pad(np.array(datum[1]),
                                pad_width=((0,MAX_SENTENCE_LENGTH-len(datum[1]))),
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

class CNN(nn.Module):
    def __init__(self, weights_matrix, hidden_size, num_classes,):

        super(CNN, self).__init__()

        self.hidden_size =  hidden_size
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix),freeze=True)
        emb_size = weights_matrix.shape[1]
        
    
        self.sent1_conv1 = nn.Conv1d(emb_size, hidden_size, kernel_size=3, padding=1)
        self.sent1_conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)

        self.sent2_conv1 = nn.Conv1d(emb_size, hidden_size, kernel_size=3, padding=1)
        self.sent2_conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)


        self.fc1 = nn.Linear(hidden_size * 2 , 300)
        self.dropout1 = nn.Dropout(0.3) 
        self.out = nn.Linear(300, num_classes)

    def forward(self, sent1, sent2):
        batch_size, seq_len = sent1.size()
        #print('batch_size1 is {}, batch_size2 is {}'.format(batch_size1, batch_size2))
        #print('seq_len1 is {}, seq_len2 is {}'.format(seq_len1,seq_len2))

        sent1_embed = self.embedding(sent1)
        sent2_embed = self.embedding(sent2)
        
        sent1_hidden = self.sent1_conv1(sent1_embed.transpose(1,2)).transpose(1,2)
        sent1_hidden = F.relu(sent1_hidden.contiguous().view(-1, sent1_hidden.size(-1))).view(batch_size, seq_len, sent1_hidden.size(-1))

        sent1_hidden = self.sent1_conv2(sent1_hidden.transpose(1,2)).transpose(1,2)
        sent1_hidden = F.relu(sent1_hidden.contiguous().view(-1, sent1_hidden.size(-1))).view(batch_size, seq_len, sent1_hidden.size(-1))

        sent1_hidden = torch.sum(sent1_hidden, dim=1)
        #print('sent1_hidden shape is {}'.format(sent1_hidden.shape)) #[32, 150]
        
        sent2_hidden = self.sent2_conv1(sent2_embed.transpose(1,2)).transpose(1,2)
        sent2_hidden = F.relu(sent2_hidden.contiguous().view(-1, sent2_hidden.size(-1))).view(batch_size, seq_len, sent2_hidden.size(-1))

        sent2_hidden = self.sent2_conv2(sent2_hidden.transpose(1,2)).transpose(1,2)
        sent2_hidden = F.relu(sent2_hidden.contiguous().view(-1, sent2_hidden.size(-1))).view(batch_size, seq_len, sent2_hidden.size(-1))

        sent2_hidden = torch.sum(sent2_hidden, dim=1)
        #print('sent2_hidden shape is {}'.format(sent2_hidden.shape)) #[32, 150]
        
        cnn_out = torch.cat([sent1_hidden, sent2_hidden], 1)
        #print('cnn out shape is {}'.format(cnn_out.shape)) #[32, 300]
        
        x = self.fc1(cnn_out)
        x = F.relu(x)
        x = self.dropout1(x)
        
        logits = self.out(x)

        return logits

### CNN:
model = CNN(weights_matrix=weights_matrix, hidden_size=450,  num_classes=3)
print(type(model))
learning_rate = 3e-4
num_epochs = 12
# number epoch to train

# Criterion and Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


#train, and save stats and best model
CNNval_accuracy = []
CNNval_loss = []
CNNtrain_accuracy = []
CNNtrain_loss = []
best_val_acc = 0
for epoch in range(num_epochs):
    for i, sample in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        # Forward pass
        outputs = model(sample[0],sample[1])
        labels = sample[2]
        loss = criterion(outputs, labels)
       
        # Backward and optimize
        loss.backward()
        optimizer.step()
        # validate every 100 iterations
        if i > 0 and i % 300 == 0:
            val_acc, val_los = test_model(model, train=False)
            tra_acc, tra_los = test_model(model, train=True)
            CNNval_accuracy.append(val_acc)
            CNNval_loss.append(val_los)
            CNNtrain_accuracy.append(tra_acc)
            CNNtrain_loss.append(tra_los)
            print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}'.format(
                       epoch+1, num_epochs, i+1, len(train_loader), val_acc))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print('find new record, save the model!')
                torch.save(model.state_dict(), 'CNN450_best_model.pkl')

OUT_DICT = {'val_accuracy':CNNval_accuracy, 'val_loss': CNNval_loss, 'train_accuracy':CNNtrain_accuracy, 'train_loss':CNNtrain_loss, 'trainable_parameters': sum([np.prod(p.size()) for p in model.parameters()if p.requires_grad ])}

pkl.dump(OUT_DICT, open("SNL_CNN_HIDDEN450.p", "wb"))
  
