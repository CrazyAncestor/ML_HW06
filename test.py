# standard library
import os
import csv
import sys
import argparse
from multiprocessing import Pool

# optional library
import jieba
import pandas as pd
from gensim.models import Word2Vec

# pytorch library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Preprocess():
    def __init__(self, data_dir, label_dir, args):
        # Load jieba library
        jieba.load_userdict(args.jieba_lib)
        self.embed_dim = args.word_dim
        self.seq_len = args.seq_len
        self.wndw_size = args.wndw
        self.word_cnt = args.cnt
        self.save_name = 'word2vec'
        self.index2word = []
        self.word2index = {}
        self.vectors = []
        self.unk = "<UNK>"
        self.pad = "<PAD>"
        # Load corpus
        if data_dir!=None:
            # Read data
            dm = pd.read_csv(data_dir)
            data = dm['comment']
            # Tokenize with multiprocessing
            # List in list out with same order
            # Multiple workers
            P = Pool(processes=4) 
            data = P.map(self.tokenize, data)
            P.close()
            P.join()
            self.data = data
            
        if label_dir!=None:
            # Read Label
            dm = pd.read_csv(label_dir)
            self.label = [int(i) for i in dm['label']]

    def tokenize(self, sentence):
        """ Use jieba to tokenize a sentence.
        Args:
            sentence (str): One string.
        Return:
            tokens (list of str): List of tokens in a sentence.
        """
        # TODO
        tokens = []
        result = jieba.tokenize(sentence, mode='search')
        for tk in result:
            tokens.append(tk[0])
        return tokens

    def get_embedding(self, load=False):
        print("=== Get embedding")
        # Get Word2vec word embedding
        if load:
            embed = Word2Vec.load(self.save_name)
        else:
            embed = Word2Vec(self.data, size=self.embed_dim, window=self.wndw_size, min_count=self.word_cnt, iter=16, workers=8)
            embed.save(self.save_name)
        # Create word2index dictinonary
        # Create index2word list
        # Create word vector list
        for i, word in enumerate(embed.wv.vocab):
            print('=== get words #{}'.format(i+1), end='\r')
            #e.g. self.word2index['魯'] = 1 
            #e.g. self.index2word[1] = '魯'
            #e.g. self.vectors[1] = '魯' vector
            self.word2index[word] = len(self.word2index)
            self.index2word.append(word)
            self.vectors.append(embed[word])
        self.vectors = torch.tensor(self.vectors)
        # Add special tokens
        self.add_embedding(self.pad)
        self.add_embedding(self.unk)
        print("=== total words: {}".format(len(self.vectors)))
        return self.vectors

    def add_embedding(self, word):
        # Add random uniform vector
        vector = torch.empty(1, self.embed_dim)
        torch.nn.init.uniform_(vector)
        self.word2index[word] = len(self.word2index)
        self.index2word.append(word)
        self.vectors = torch.cat([self.vectors, vector], 0)

    def get_indices(self,test=False):
        # Transform each words to indices
        # e.g. if 機器=0,學習=1,好=2,玩=3 
        # [機器,學習,好,好,玩] => [0, 1, 2, 2,3]
        all_indices = []
        # Use tokenized data
        for i, sentence in enumerate(self.data):
            print('=== sentence count #{}'.format(i+1), end='\r')
            sentence_indices = []
            for word in sentence:
                # if word in word2index append word index into sentence_indices
                # if word not in word2index append unk index into sentence_indices
                # TODO
                flag=0
                                
                for wd in self.index2word:
                    if wd == word:
                        flag=1
                if flag==1:
                    sentence_indices.append(self.word2index[word])
                else :
                    w="<UNK>"
                    sentence_indices.append(self.word2index[w])
            # pad all sentence to fixed length
            sentence_indices = self.pad_to_len(sentence_indices, self.seq_len, self.word2index[self.pad])
            all_indices.append(sentence_indices)
        if test:
            return torch.LongTensor(all_indices)         
        else:
            return torch.LongTensor(all_indices), torch.LongTensor(self.label)        

    def pad_to_len(self, arr, padded_len, padding=0):
        """ 
        if len(arr) < padded_len, pad arr to padded_len with padding.
        If len(arr) > padded_len, truncate arr to padded_len.
        Example:
            pad_to_len([1, 2, 3], 5, 0) == [1, 2, 3, 0, 0]
            pad_to_len([1, 2, 3, 4, 5, 6], 5, 0) == [1, 2, 3, 4, 5]
        Args:
            arr (list): List of int.
            padded_len (int)
            padding (int): Integer used to pad.
        Return:
            arr (list): List of int with size padded_len.
        """
        
        if len(arr)<= padded_len:
            delta = padded_len-len(arr)
            for i in range(delta):
                arr.append(padding)
        else:
            delta = len(arr)-padded_len
            for i in range(delta):
                arr.pop()
        return arr
        # TODO

class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_emb=True):
        super(LSTM_Net, self).__init__()
        # Create embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # Fix/Train embedding 
        self.embedding.weight.requires_grad = False if fix_emb else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid())
    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None)
        # x dimension(batch, seq_len, hidden_size)
        # Use LSTM last hidden state (maybe we can use more states)
        x = x[:, -1, :] 
        x = self.classifier(x)
        return x

def evaluation(outputs, labels):
    #outputs => probability (float)
    #labels => labels
    outputs[outputs>=0.5] = 1
    outputs[outputs<0.5] = 0
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct

def training(args, train, valid, model, device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\n=== start training, parameter total:{}, trainable:{}'.format(total, trainable))
    model.train()
    batch_size, n_epoch = args.batch, args.epoch
    criterion = nn.BCELoss()
    t_batch = len(train) 
    v_batch = len(valid) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    total_loss, total_acc, best_acc = 0, 0, 0
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        # training set
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            correct = evaluation(outputs, labels)
            total_acc += (correct / batch_size)
            total_loss += loss.item()
            print('[ Epoch{} == {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
            	epoch+1, i+1, t_batch, loss.item(), correct*100/batch_size), end='\r')
        print('\nTrain | Loss:{:.5f} Acc: {:.3f} '.format(total_loss/t_batch, total_acc/t_batch*100))

        # validation set
        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.float)
                outputs = model(inputs)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                correct = evaluation(outputs, labels)
                total_acc += (correct / batch_size)
                total_loss += loss.item()

            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss/v_batch, total_acc/v_batch*100))
            if total_acc > best_acc:
                 best_acc = total_acc
                 torch.save(model, "{}/ckpt_{:.3f}".format(args.model_dir,total_acc/v_batch*100))
                 print('saving model with acc {:.3f}'.format(total_acc/v_batch*100))
        model.train()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = Preprocess(args.train_X, args.train_Y, args)
    # Get word embedding vectors
    embedding = preprocess.get_embedding(load=False)
    # Get word indices
    data, label = preprocess.get_indices()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str, help='[Output] Your model checkpoint directory')
    parser.add_argument('jieba_lib',type=str, help='[Input] Your jieba dict.txt.big')
    parser.add_argument('train_X',type=str, help='[Input] Your train_x.csv')
    parser.add_argument('train_Y',type=str, help='[Input] Your train_y.csv')

    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch', default=128, type=int)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--seq_len', default=30, type=int)
    parser.add_argument('--word_dim', default=100, type=int)
    parser.add_argument('--hidden_dim', default=100, type=int)
    parser.add_argument('--wndw', default=3, type=int)
    parser.add_argument('--cnt', default=3, type=int)
    args = parser.parse_args()
    main(args)