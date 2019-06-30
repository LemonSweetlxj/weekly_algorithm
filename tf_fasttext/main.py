import numpy as np
import os,time,sys
import tensorflow as tf
from utils import get_logger,str2bool
import datetime
import pickle
import codecs
from base_function import read_corpus, read_dictionary, random_embedding
from model import FASTTEXT
import argparse

tag2label = {"1": [1,0,0,0], "2": [0,1,0,0],"3":[0,0,1,0],"4":[0,0,0,1]}

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2

parser = argparse.ArgumentParser(description='FASTTEXT for intent classify')
parser.add_argument('--mode', type=str, default='train', help='train/test')
parser.add_argument('--demo_model',type=str,default='6.0',help='model for test and demo')
parser.add_argument('--train_data', type=str, default='data_path', help='train data source')
parser.add_argument('--test_data', type=str, default='data_path', help='test data source')
parser.add_argument('--train_file', type=str, default='train_data', help='train data source')
parser.add_argument('--test_file', type=str, default='test_data', help='test data source')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--embedding_dim',type=int,default=300,help='random init char embedding_dim')
parser.add_argument('--pretrain_embedding',type=str,default='random',help='use pretrained char embedding or init it randomly')
parser.add_argument('--update_embedding',type=str2bool,default=True,help='update embedding during training')
parser.add_argument('--batch_size', type=int, default=20, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=10, help='#epoch of training')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.8, help='dropout keep_prob')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--shuffle',type=str2bool,default=True,help='shuffle training data before each epoch')
args = parser.parse_args()

# path setting
paths = {}
nowtime = datetime.datetime.now().strftime('%m-%d_%H_%M')
timestamp = nowtime if args.mode == 'train' else args.demo_model
output_path = os.path.join('.', args.train_data + "_save", timestamp)
if not os.path.exists(output_path):os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path):os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path):os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path):os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))

## get char embeddings
word2id = read_dictionary(os.path.join('.', args.train_data, 'word.pkl'))
if args.pretrain_embedding == 'random':
    # 随机初始化词向量
    embeddings = random_embedding(word2id, args.embedding_dim)
else:
    #预训练词向量
    embedding_path = codecs.open('./data_path/query.vec','r','utf-8')
    all_embedding = []
    UNK_embedding = (np.float32(np.random.uniform(-0.25, 0.25, (1, 200)))).tolist()
    PAD_embedding = (np.float32(np.random.uniform(-0.25, 0.25, (1, 200)))).tolist()
    for line in embedding_path:
        each_embedding = []
        temp = str(line).strip('\n').split()
        for i in temp[1:]:
            each_embedding.append(float(i))
        all_embedding.append(each_embedding)
    all_embedding.extend(UNK_embedding)
    all_embedding.extend(PAD_embedding)
    embeddings = np.array(all_embedding)
    print(embeddings.shape)

train_path = os.path.join('.', args.train_data, args.train_file)
test_path = os.path.join('.', args.train_data, args.train_file)
train_data = read_corpus(train_path)
test_data = read_corpus(test_path)
print(test_data[0])

## train mode
if args.mode == "train":
    model = FASTTEXT(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    print("train data:{}".format(len(train_data)))
    model.train(train = train_data, dev = test_data)

##test / predicta
if args.mode == "test":
    timeNow = time.strftime('%m-%d-%H%M', time.localtime(time.time()))
    f1 = codecs.open('./predict/predict_res_{}'.format(timeNow), 'w', encoding='utf-8')
    model_path = '././data_path_save/saved_model/{}'.format(args.demo_model)
    paths['model_path'] = model_path
    print("test data: {}" .format(len(test_data)))
    model = FASTTEXT(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    all_label,all_out = model.test(sess,test_data)
    count = 0
    for index,(query,doc,tag) in enumerate(test_data):
        if int(tag[0]) != all_out[index]:
            f1.write(''.join(query) + '\t' + ''.join(doc) + '\t' + tag[0] + '\t' + str(all_out[index]) + '\t' + str(all_label[index][0]) + '\n')
            if all_label[index][0] < 0:
                count += 1
    print(count)
