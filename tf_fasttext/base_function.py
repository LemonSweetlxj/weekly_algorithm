import sys, pickle, os, random
import numpy as np

# 读字典
def read_dictionary(vocab_path):
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id

# embedding
def random_embedding(vocab, embedding_dim):
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat

#读语料
def read_corpus(corpus_path):
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    query_,tag_ = [],[]
    for line in lines:
        [query,label] = str(line).strip('\n').split('\t')
        for char in query:
            query_.append(char)
        tag_.append(label)
        data.append((query_,tag_))
        query_,tag_ = [],[]
    return data

# 语料转id 
def sentence2id(sent, word2id):
    sentence_id = []
    for word in sent:
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id

# 补0
def pad_sequences(sequences, pad_mark=0):
    max_len = max(map(lambda x: len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list

# batch
def batch_yield(data, batch_size, vocab, tag2label,shuffle=False):
    if shuffle:
        random.shuffle(data)
    queries,labels = [],[]
    for (queries_,tag_) in data:
        queries_ = sentence2id(queries_, vocab)
        label_ = tag2label[tag_[0]] 

        if len(queries) == batch_size:
            yield queries, labels
            queries, labels = [],[]
        
        queries.append(queries_)
        labels.append(label_)
    
    if len(queries) != 0:
        yield queries, labels







