import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
import sys
import time
from sklearn.metrics import f1_score
import random

class cnn(object):

    def __init__(self,embedding_matrix,num_classes,max_words,
                 num_filters=300,dropout_keep=0.5,lr=0.0002,l2norm=False):

        self.dropout_keep = dropout_keep
        self.dropout = tf.placeholder(tf.float32)
        self.mw = max_words
        if l2norm:
            self.initializer = tf.orthogonal_initializer()
        else:
            self.initializer = tf.contrib.layers.xavier_initializer()

        #doc input and embeddings
        self.doc_input = tf.placeholder(tf.int32, shape=[None,max_words])
        embeddings = tf.get_variable('embeddings',initializer=
                     embedding_matrix.astype(np.float32),dtype=tf.float32)
        if l2norm:
            embeddings = tf.nn.l2_normalize(embeddings,1)
        word_embeds = tf.gather(embeddings,self.doc_input)

        #word convolutions
        conv3 = tf.layers.conv1d(word_embeds,num_filters,3,padding='same',
                activation=tf.nn.relu,kernel_initializer=self.initializer)
        conv4 = tf.layers.conv1d(word_embeds,num_filters,4,padding='same',
                activation=tf.nn.relu,kernel_initializer=self.initializer)
        conv5 = tf.layers.conv1d(word_embeds,num_filters,5,padding='same',
                activation=tf.nn.relu,kernel_initializer=self.initializer)
        pool3 = tf.reduce_max(conv3,1)
        pool4 = tf.reduce_max(conv4,1)
        pool5 = tf.reduce_max(conv5,1)

        #concatenate
        concat = tf.concat([pool3,pool4,pool5],1)
        doc_embed = tf.nn.dropout(concat,self.dropout)

        #classification functions
        output = tf.layers.dense(doc_embed,num_classes,
                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.prediction = tf.nn.softmax(output)

        #loss, accuracy, and training functions
        self.labels = tf.placeholder(tf.int32)
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

    def train(self,data,labels,batch_size=16,epochs=10,patience=3,validation_data=None,profile=False):

        if validation_data:
            validation_size = len(validation_data[0])
        else:
            validation_size = len(data)

        print('training network on %i documents, validation on %i documents' \
              % (len(data), validation_size))

        #track best model for saving
        prevbest = 0
        pat_count = 0

        for ep in range(epochs):

            #shuffle data
            xy = list(zip(data,labels))
            random.shuffle(xy)
            data,labels = zip(*xy)
            data = list(data)
            labels = list(labels)

            y_pred = []
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
                    with open('profile/cnn_step%i.json' % start, 'w') as f:
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
    max_words = 1500
    num_classes = 10
    embedding_size = 300

    #create data
    vocab = np.random.rand(vocab_size,embedding_size)
    X = np.random.randint(0,vocab_size,(samples,max_words))
    y = np.random.randint(0,num_classes,samples)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,stratify=y)

    if not os.path.exists('profile'):
         os.makedirs('profile')

    #train cnn
    model = cnn(vocab,num_classes,1500,lr=lr)
    model.train(X_train,y_train,epochs=epochs,batch_size=batch_size,validation_data=(X_test,y_test),profile=True)
