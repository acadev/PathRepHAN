import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
import sys
import time
from sklearn.metrics import f1_score
import random

class hisan(object):

    def __init__(self,embedding_matrix,num_classes,max_sents,max_words,attention_heads=8,
                 attention_size=512,dropout_keep=0.9,activation=tf.nn.elu,lr=0.00005):

        self.dropout_keep = dropout_keep
        self.dropout = tf.placeholder(tf.float32)
        self.ms = max_sents
        self.mw = max_words
        self.embedding_matrix = embedding_matrix.astype(np.float32)
        self.attention_size = attention_size
        self.attention_heads = attention_heads
        self.activation = activation

        #doc input
        self.doc_input = tf.placeholder(tf.int32, shape=[None,max_sents,max_words])
        self.doc_embeds = tf.map_fn(self._attention_step,self.doc_input,dtype=tf.float32)

        #classification functions
        output = tf.layers.dense(self.doc_embeds,num_classes,
                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.prediction = tf.nn.softmax(output)

        #loss, accuracy, and training functions
        self.labels = tf.placeholder(tf.int32,shape=[None])
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,labels=self.labels))
        self.optimizer = tf.train.AdamOptimizer(lr,0.9,0.99).minimize(self.loss)

        #init op
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.saver = tf.train.Saver()
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self.run_metadata = tf.RunMetadata()

    def _attention_step(self,doc):

        words_per_line = tf.reduce_sum(tf.sign(doc),1)
        num_lines = tf.reduce_sum(tf.sign(words_per_line))
        max_words_ = tf.reduce_max(words_per_line)
        doc_input_reduced = doc[:num_lines,:max_words_]
        num_words = words_per_line[:num_lines]

        #word embeddings
        word_embeds = tf.gather(tf.get_variable('embeddings',initializer=self.embedding_matrix,
                      dtype=tf.float32),doc_input_reduced)
        word_embeds = tf.nn.dropout(word_embeds,self.dropout)

        #masking
        mask_base = tf.cast(tf.sequence_mask(num_words,max_words_),tf.float32)
        mask = tf.tile(tf.expand_dims(mask_base,2),[1,1,self.attention_size])
        mask2 = tf.tile(tf.expand_dims(mask_base,2),[self.attention_heads,1,max_words_])

        #word self attention
        Q = tf.layers.conv1d(word_embeds,self.attention_size,1,padding='same',
            activation=self.activation,kernel_initializer=tf.contrib.layers.xavier_initializer())
        K = tf.layers.conv1d(word_embeds,self.attention_size,1,padding='same',
            activation=self.activation,kernel_initializer=tf.contrib.layers.xavier_initializer())
        V = tf.layers.conv1d(word_embeds,self.attention_size,1,padding='same',
            activation=self.activation,kernel_initializer=tf.contrib.layers.xavier_initializer())

        Q = tf.where(tf.equal(mask,0),tf.zeros_like(Q),Q)
        K = tf.where(tf.equal(mask,0),tf.zeros_like(K),K)
        V = tf.where(tf.equal(mask,0),tf.zeros_like(V),V)

        Q_ = tf.concat(tf.split(Q,self.attention_heads,axis=2),axis=0)
        K_ = tf.concat(tf.split(K,self.attention_heads,axis=2),axis=0)
        V_ = tf.concat(tf.split(V,self.attention_heads,axis=2),axis=0)

        outputs = tf.matmul(Q_,tf.transpose(K_,[0, 2, 1]))
        outputs = outputs/(K_.get_shape().as_list()[-1]**0.5)
        outputs = tf.where(tf.equal(outputs,0),tf.ones_like(outputs)*-1000,outputs)
        outputs = tf.nn.dropout(tf.nn.softmax(outputs),self.dropout)
        outputs = tf.where(tf.equal(mask2,0),tf.zeros_like(outputs),outputs)
        outputs = tf.matmul(outputs,V_)
        outputs = tf.concat(tf.split(outputs,self.attention_heads,axis=0),axis=2)
        outputs = tf.where(tf.equal(mask,0),tf.zeros_like(outputs),outputs)

        #word target attention
        Q = tf.get_variable('word_Q',(1,1,self.attention_size),
            tf.float32,tf.orthogonal_initializer())
        Q = tf.tile(Q,[num_lines,1,1])

        Q_ = tf.concat(tf.split(Q,self.attention_heads,axis=2),axis=0)
        K_ = tf.concat(tf.split(outputs,self.attention_heads,axis=2),axis=0)
        V_ = tf.concat(tf.split(outputs,self.attention_heads,axis=2),axis=0)

        outputs = tf.matmul(Q_,tf.transpose(K_,[0, 2, 1]))
        outputs = outputs/(K_.get_shape().as_list()[-1]**0.5)
        outputs = tf.where(tf.equal(outputs,0),tf.ones_like(outputs)*-1000,outputs)
        outputs = tf.nn.dropout(tf.nn.softmax(outputs),self.dropout)
        outputs = tf.matmul(outputs,V_)
        outputs = tf.concat(tf.split(outputs,self.attention_heads,axis=0),axis=2)
        sent_embeds = tf.transpose(outputs,[1,0,2])
        sent_embeds = tf.nn.dropout(sent_embeds,self.dropout)

        #sent self attention
        Q = tf.layers.conv1d(sent_embeds,self.attention_size,1,padding='same',
            activation=self.activation,kernel_initializer=tf.contrib.layers.xavier_initializer())
        K = tf.layers.conv1d(sent_embeds,self.attention_size,1,padding='same',
            activation=self.activation,kernel_initializer=tf.contrib.layers.xavier_initializer())
        V = tf.layers.conv1d(sent_embeds,self.attention_size,1,padding='same',
            activation=self.activation,kernel_initializer=tf.contrib.layers.xavier_initializer())

        Q_ = tf.concat(tf.split(Q,self.attention_heads,axis=2),axis=0)
        K_ = tf.concat(tf.split(K,self.attention_heads,axis=2),axis=0)
        V_ = tf.concat(tf.split(V,self.attention_heads,axis=2),axis=0)

        outputs = tf.matmul(Q_,tf.transpose(K_,[0, 2, 1]))
        outputs = outputs/(K_.get_shape().as_list()[-1]**0.5)
        outputs = tf.nn.dropout(tf.nn.softmax(outputs),self.dropout)
        outputs = tf.matmul(outputs,V_)
        outputs = tf.concat(tf.split(outputs,self.attention_heads,axis=0),axis=2)

        #sent target attention
        Q = tf.get_variable('sent_Q',(1,1,self.attention_size),
            tf.float32,tf.orthogonal_initializer())

        Q_ = tf.concat(tf.split(Q,self.attention_heads,axis=2),axis=0)
        K_ = tf.concat(tf.split(outputs,self.attention_heads,axis=2),axis=0)
        V_ = tf.concat(tf.split(outputs,self.attention_heads,axis=2),axis=0)

        outputs = tf.matmul(Q_,tf.transpose(K_,[0, 2, 1]))
        outputs = outputs/(K_.get_shape().as_list()[-1]**0.5)
        outputs = tf.nn.dropout(tf.nn.softmax(outputs),self.dropout)
        outputs = tf.matmul(outputs,V_)
        outputs = tf.concat(tf.split(outputs,self.attention_heads,axis=0),axis=2)
        doc_embed = tf.nn.dropout(tf.squeeze(outputs,[0]),self.dropout)
        doc_embed = tf.squeeze(doc_embed,[0])

        return doc_embed

    def train(self,data,labels,batch_size=16,epochs=30,patience=5,validation_data=None,profile=False):

        if validation_data:
            validation_size = len(validation_data[0])
        else:
            validation_size = len(data)

        print('training network on %i documents, validation on %i documents' \
              % (len(data), validation_size))

        for ep in range(epochs):

            #shuffle data
            xy = list(zip(data,labels))
            random.shuffle(xy)
            data,labels = zip(*xy)
            data = list(data)
            labels = list(labels)

            y_pred = []
            y_true = []
            start_time = time.time()

            #train
            for start in range(0,len(data),batch_size):

                #get batch index
                if start+batch_size < len(data):
                    stop = start+batch_size
                else:
                    stop = len(data)

                feed_dict = {self.doc_input:data[start:stop],self.labels:labels[start:stop],self.dropout:self.dropout_keep}

                if profile:
                    pred,cost,_ = self.sess.run([self.prediction,self.loss,self.optimizer],feed_dict=feed_dict,
                                  options=self.options,run_metadata=self.run_metadata)
                    fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open('profile/hisan_step%i.json' % start, 'w') as f:
                        f.write(chrome_trace)

                else:
                    pred,cost,_ = self.sess.run([self.prediction,self.loss,self.optimizer],feed_dict=feed_dict)

                #track correct predictions
                y_pred.append(np.argmax(pred,1))
                sys.stdout.write("epoch %i, sample %i of %i, loss: %f        \r"\
                                 % (ep+1,stop+1,len(data),cost))
                sys.stdout.flush()

            #checkpoint after every epoch
            print("\ntraining time: %.2f" % (time.time()-start_time))

            y_pred = np.concatenate(y_pred,0)
            micro = f1_score(labels,y_pred,average='micro')
            macro = f1_score(labels,y_pred,average='macro')
            print("epoch %i training micro/macro: %.4f, %.4f" % (ep+1,micro,macro))

            micro,macro = self.score(validation_data[0],validation_data[1],batch_size=batch_size)
            print("epoch %i validation micro/macro: %.4f, %.4f" % (ep+1,micro,macro))

            #reset timer
            start_time = time.time()

    def predict(self,data,batch_size=16):

        y_pred = []
        for start in range(0,len(data),batch_size):

            #get batch index
            if start+batch_size < len(data):
                stop = start+batch_size
            else:
                stop = len(data)

            feed_dict = {self.doc_input:data[start:stop],self.dropout:1.0}
            prob = self.sess.run(self.prediction,feed_dict=feed_dict)
            y_pred.append(np.argmax(prob,1))

            sys.stdout.write("processed %i of %i records        \r" % (stop+1,len(data)))
            sys.stdout.flush()

        print()
        y_pred = np.concatenate(y_pred,0)
        return y_pred

    def score(self,data,labels,batch_size=16):

        y_pred = self.predict(data,batch_size)
        micro = f1_score(labels,y_pred,average='micro')
        macro = f1_score(labels,y_pred,average='macro')
        return micro,macro

    def save(self,filename):
        self.saver.save(self.sess,filename)

    def load(self,filename):
        self.saver.restore(self.sess,filename)

if __name__ == "__main__":

    import pickle
    from sklearn.model_selection import train_test_split

    #params
    batch_size = 64
    lr = 0.0001
    epochs = 10
    samples = 50000
    vocab_size = 10000
    max_lines = 150
    max_words = 30
    num_classes = 10
    embedding_size = 300

    #create data
    vocab = np.random.rand(vocab_size,embedding_size)
    X = np.random.randint(0,vocab_size,(samples,max_lines,max_words))
    y = np.random.randint(0,num_classes,samples)

    #optional masking
    min_lines = 30
    min_words = 5
    mask = []
    for i in range(samples):
        doc_mask = np.ones((1,max_lines,max_words))
        num_lines = np.random.randint(min_lines,max_lines)
        for j in range(num_lines):
            num_words = np.random.randint(min_words,max_words)
            doc_mask[0,j,:num_words] = 0
        mask.append(doc_mask)

    mask = np.concatenate(mask,0)
    X[mask.astype(np.bool)] = 0

    #test train split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,stratify=y)

    if not os.path.exists('profile'):
         os.makedirs('profile')

    #train hcan
    model = hisan(vocab,num_classes,max_lines,max_words,lr=lr)
    model.train(X_train,y_train,epochs=epochs,batch_size=batch_size,validation_data=(X_test,y_test),profile=True)
