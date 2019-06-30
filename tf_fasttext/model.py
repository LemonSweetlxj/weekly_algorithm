import os, time,sys
import tensorflow as tf
from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.contrib.rnn import LSTMCell
import math
import datetime
from base_function import pad_sequences,batch_yield
from utils import get_logger
import numpy as np
from tensorflow.contrib import slim

class FASTTEXT(object):
    def __init__(self,args,embeddings,tag2label,word2id,paths,config):
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.hidden_dim = args.hidden_dim
        self.embeddings = embeddings
        self.update_embedding = args.update_embedding
        self.clip_grad = args.clip
        self.vocab = word2id
        self.learning_rate = args.lr # 具体值
        self.dropout = args.dropout # 具体值
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.shuffle = args.shuffle
        self.model_path = paths['model_path']
        self.summary_path = paths['summary_path']
        self.logger = get_logger(paths['log_path'])
        self.result_path = paths['result_path']
        self.config = config


    def build_graph(self):
        # 占位符
        self.input_data = tf.placeholder(tf.int32, shape = [None,None],name = "input_data")
        self.sequence_lengths = tf.placeholder(dtype=tf.int32, shape=[None],name = "sequence_lengths")
        self.labels = tf.placeholder(dtype = tf.int32,shape = [None,self.num_tags],name = "labels")
        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
        
        # 词向量映射
        _word_embeddings = tf.Variable(self.embeddings,dtype=tf.float32,trainable=self.update_embedding,name="word_embeddings")
        self.embed_queries = tf.nn.dropout(tf.nn.embedding_lookup(params=_word_embeddings,ids=self.input_data,name="query_embeddings"),self.dropout_pl)
        
        # 对词向量进行平均
        with tf.variable_scope('average'):
            mean_sentence = tf.reduce_mean(self.embed_queries,axis = 1)
        
        # 输出层
        with tf.varialbe_scope('score'):
            self.logits = tf.layers.dense(mean_sentence,self.num_tags,name = 'dense_layer')
            self.pred_y = tf.argmax(tf.nn.softmax(self.logits),1,name = "pred")
        # 损失函数
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels),name = "loss")
       
        # 优化函数
        self.global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(self.lr_pl,self.global_step,decay_steps=100,decay_rate=0.9,staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rage)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.optim = slim.learning.create_train_op(total_loss=self.loss, optimizer=optimizer,update_ops=update_ops)
        
        # 准确率
        self.label_y = tf.argmax(self.label,1)
        self.pred = tf.equal(self.pred_y,self.label_y)
        self.accuracy = tf.reduce_mean(tf.cast(self.pred, tf.float32),name = "accuracy")
        tf.summary.scalar("loss",self.loss)
    
    def add_summary(self, sess):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def get_feed_dict(self,queries,labels=None, lr=None,dropout = None):
        index_queries,queries_len_list = pad_sequences(queries,pad_mark=0)
        feed_dict = {self.input_data:index_queries,self.sequence_lengths:queries_len_list}
        if labels is not None:
            feed_dict[self.labels] = labels
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout
        return feed_dict

    # 跑一轮
    def run_one_epoch(self,sess,train,dev,tag2label,epoch,saver):
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        batches = batch_yield(train,self.batch_size,self.vocab,self.tag2label,shuffle=self.shuffle)
        
        for step, (queries, labels) in enumerate(batches):
            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            step_num = epoch * num_batches + step + 1
            feed_dict = self.get_feed_dict(queries,labels, self.learning_rate,self.dropout)
            print(feed_dict)
            loss_train,accuracy,summary = sess.run([self.loss, self.accuracy, self.merged],feed_dict = feed_dict)
            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                self.logger.info('{} epoch {}, step {}, loss: {:.4},accuracy:{:.4}, global_step: {}'.\
                        format(start_time, epoch + 1, step + 1, loss_train,accuracy,step_num))
            self.file_writer.add_summary(summary, step_num)

            if step + 1 == num_batches:
                saver.save(sess, self.model_path, global_step=step_num)

        self.logger.info('===========validation / test===========')
        self.dev_one_epoch(sess, dev)
    
    # 计算测评数据
    def tf_confusion_metrics(self,predict,real,sess,feed_dict):
        ones_like_actuals = tf.ones_like(real)
        zeros_like_actuals = tf.zeros_like(real)
        ones_like_predictions = tf.ones_like(predict)
        zeros_like_predictions = tf.zeros_like(predict)

        tp_op = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(real, ones_like_actuals),tf.equal(predict, ones_like_predictions)),"float" ))
        tn_op = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(real, zeros_like_actuals),tf.equal(predict, zeros_like_predictions)),"float" ))
        fp_op = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(real, zeros_like_actuals),tf.equal(predict, ones_like_predictions)),"float" ))
        fn_op = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(real, ones_like_actuals),tf.equal(predict, zeros_like_predictions)),"float" ))

        tp, tn, fp, fn = sess.run([tp_op, tn_op, fp_op, fn_op], feed_dict)
        recall = float(tp)/(float(tp) + float(fn) + 0.0001)
        accuracy = (float(tp) + float(tn))/(float(tp) + float(fp) + float(fn) + float(tn) + 0.0001)
        precision = float(tp)/(float(tp) + float(fp) + 0.0001)
        f1_score = (2 * (precision * recall)) / (precision + recall + 0.0001)
        return accuracy,recall,precision,f1_score
   
   # 测一轮
    def dev_one_epoch(self,sess, dev):
        all_accuracy,all_recall,all_precision,all_f1 = [],[],[],[]
        all_label = []
        num_batches = (len(dev) + self.batch_size - 1) // self.batch_size
        batches = batch_yield(dev,self.batch_size,self.vocab,self.tag2label,shuffle=self.shuffle)
        for step,(queries, labels) in enumerate(batches):
            feed_dict = self.get_feed_dict(queries,labels, dropout = 1.0)
            label_list,test_loss,test_accuracy = sess.run([self.pred,self.loss,self.accuracy],feed_dict)
            #print(q_state)
            #print(d_state)
            all_label.extend(label_list)

            real_tags = tf.argmax(labels,1)
            accu,recall,precision,f1_score = self.tf_confusion_metrics(label_list, real_tags, sess, feed_dict)
            if accu != 0 : all_accuracy.append(accu)
            if recall != 0 :all_recall.append(recall)
            if precision !=0:all_precision.append(precision)
            if f1_score !=0:all_f1.append(f1_score)

        if len(all_accuracy) != 0: print("average accuracy:",sum(all_accuracy)/len(all_accuracy))
        if len(all_recall) != 0:print("average recall:",sum(all_recall)/len(all_recall))
        if len(all_precision) != 0:print("average precision:",sum(all_precision)/len(all_precision))
        if len(all_f1) != 0:print("average f1:",sum(all_f1)/len(all_f1))

        return all_label
    
    # 模型训练
    def train(self, train,dev):
        saver = tf.train.Saver(tf.global_variables())
        Time = datetime.datetime.now().strftime('%m-%d_%H_%M')
        with tf.Session(config=self.config) as sess:
            sess.run(tf.global_variables_initializer())
            self.add_summary(sess)

            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train, dev, self.tag2label, epoch,saver)
            export_path = os.path.join('./data_path_save/saved_model/', Time)
            builder = tf.saved_model.builder.SavedModelBuilder(export_path)
            input_save = {"word_ids":tf.saved_model.utils.build_tensor_info(self.input_data),\
                          "labels":tf.saved_model.utils.build_tensor_info(self.labels),   
                          "sequence_lengths":tf.saved_model.utils.build_tensor_info(self.sequence_lengths),\
                          "dropout":tf.saved_model.utils.build_tensor_info(self.dropout_pl)}
            output_save ={"outputs": tf.saved_model.utils.build_tensor_info(self.pred)}
            signature = tf.saved_model.signature_def_utils.build_signature_def(input_save, output_save, "tensorflow/serving/predict")
            builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],{
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:signature})
            builder.save()


    def test(self,sess,dev):
        session = tf.Session(graph = tf.Graph())
        meta_graph = tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.SERVING], self.model_path)
        model_graph_signature = list(meta_graph.signature_def.items())[0][1]
        output_tensor_names = []
        output_op_names = []
        for output_item in model_graph_signature.outputs.items():
            output_op_name = output_item[0]
            output_op_names.append(output_op_name)
            output_tensor_name = output_item[1].name
            output_tensor_names.append(output_tensor_name)
        sentences = {}
        all_accuracy,all_recall,all_precision,all_f1 = [],[],[],[]
        num_batches = (len(dev) + self.batch_size - 1) // self.batch_size
        all_label,all_out = [],[]
        batches = batch_yield(dev,self.batch_size,self.vocab,self.tag2label,shuffle=self.shuffle)
        for step,(queries, docs, labels) in enumerate(batches):
            index_queries,queries_len_list = pad_sequences(queries,pad_mark=0)
            sentences["index_queries"] = index_queries
            sentences["index_docs"] = index_docs
            feed_dict_map = {}

            for input_item in model_graph_signature.inputs.items():
                input_op_name = input_item[0]
                input_tensor_name = input_item[1].name
                feed_dict_map[input_tensor_name] = sentences[input_op_name]
            sim_list = session.run(output_tensor_names,feed_dict_map)
            sim_list = sim_list[0].tolist()
            unsim_list = []
            for each_sim_list in sim_list:
                unsim_list.append([0-each_sim_list])
            sim_list = np.expand_dims(sim_list, axis=1)
            out = np.concatenate([unsim_list,sim_list],axis = 1)  
            all_label.extend(sim_list)

            pred = tf.argmax(out,1)
            tag = tf.argmax(labels,1)
            _pred = tf.Session().run(pred)
            all_out.extend(_pred)
            
            accu,recall,precision,f1_score = self.tf_confusion_metrics(pred, tag, sess, feed_dict_map)
            if accu != 0 : all_accuracy.append(accu)
            if recall != 0 :all_recall.append(recall)
            if precision !=0:all_precision.append(precision)
            if f1_score !=0:all_f1.append(f1_score)

        if len(all_accuracy) != 0: print("average accuracy:",sum(all_accuracy)/len(all_accuracy))
        if len(all_recall) != 0:print("average recall:",sum(all_recall)/len(all_recall))
        if len(all_precision) != 0:print("average precision:",sum(all_precision)/len(all_precision))
        if len(all_f1) != 0:print("average f1:",sum(all_f1)/len(all_f1))
        
        return all_label,all_out


    
