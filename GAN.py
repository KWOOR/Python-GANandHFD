# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 21:58:15 2019

@author: kur7
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sa
#import sys
#from sklearn.decomposition import PCA
#import statsmodels.api as sm
#import time
#from tqdm import tqdm
#from sklearn.model_selection import train_test_split
#import xgboost as xgb
import os
import tensorflow as tf
from numpy import math
os.chdir('C:\\Users\\kur7\\OneDrive\\바탕 화면\\uiuc\\DA\\Daily')

def make_name(data):
    name=dict(zip(range(len(data)),data.keys()))
    name_buff=dict(zip(data.keys(),range(len(data))))
    return name, name_buff

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # zero division 에러를 해결하기 위해 분모에 아주 작은 값을 더해줌
    return numerator / (denominator + 1e-7)
#%%

df1 = pd.read_csv('Big_std_final_data1.csv').iloc[:,1:]
df2 = pd.read_csv('Big_std_final_data2.csv').iloc[:,1:]

real_final = pd.concat([df1,df2])

del df1, df2

df_final=dict(list(real_final.groupby(real_final['Ticker'])))
del real_final

name,_ = make_name(df_final)

#%%

data = df_final[name[9]]



rolling_window = 3 #5일씩 롤링
train_period = 20 #20일 동안 학습하고
test_period = 5 #5일동안 테스트
pred_days = 10

trainX = []
trainY = []
ticker=data.columns[2:-1]

for i in range(0,len(data)-pred_days):
    trainX.append(data.iloc[i:i+pred_days][ticker].values)
    trainY.append(data.iloc[i+pred_days]['Price'])    

testX = []
testY = []
for i in range(0,len(data)-pred_days-pred_days) :
    testX.append(data.iloc[i+pred_days:i+pred_days+pred_days][ticker].values)
    testY.append(data.iloc[i+pred_days+pred_days]['Price'])      
    

del data
#%%
def gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * math.pow(x, 3))))



seq_length = 20
input_data_column_num = 19
learning_rate = 0.0004
num_stacked_layers = 5
num_units = 120
keep_prob = 0.8
iterations = 20



#%%

''' 트레인 셋 따로, 테스트 셋 따로 민맥스 스케일링을 한다! PCA에서는 스탠다드가 좋고 일반 뉴럴넷에서는 민맥스가 좋음'''
''' 따라서.. 뉴럴넷을 돌리기 위해 민맥스 스케일링을 하는것임! '''

def refine_data(data):
    buff = pd.DataFrame()
    for i in range(len(data)):
        buff = pd.concat( [buff, pd.DataFrame(data[i])], axis=0)
    return buff.values

trainX = refine_data(trainX)
trainY = trainY
testX = refine_data(testX)
testY = testY

#def refine_data(data):
#    buff = pd.DataFrame()
#    for i in range(len(data)):
#        buff = pd.concat( [buff, pd.DataFrame(data[i])], axis=0)
#    return MinMaxScaler(buff).values
#
#trainX = refine_data(trainX)
#trainY = MinMaxScaler(trainY)
#testX = refine_data(testX)
#testY = MinMaxScaler(testY)


#%%

nDataRow_E = 10 #10일치 넣고 학습..
nDataCol = trainX.shape[1] #19
nGInput = 20
nGHidden = 128
nDHidden = 128


def LSTM_cell(keep_prob):
    # forget_bias: biases of the forget gate (default: 1) in order to reduce the scale of forgetting in the beginning of the training.
    
    cell = tf.contrib.rnn.LSTMCell(num_units= num_units,state_is_tuple=True,
                                   initializer= tf.contrib.layers.xavier_initializer())
    if keep_prob < 1.0:
         cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob, state_keep_prob= keep_prob)
    return cell


def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        # num_stacked_layers개의 층으로 쌓인 Stacked RNNs 생성
        stackedRNNs = [LSTM_cell(keep_prob) for _ in range(num_stacked_layers)]
        multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs, state_is_tuple=True) 
        
        # RNN Cell(여기서는 LSTM셀임)들을 연결
        outputs, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)        
        # [:, -1]를 잘 살펴보자. LSTM RNN의 마지막 (hidden)출력만을 사용했다.
        # 과거 여러 거래일의 주가를 이용해서 다음날의 주가 1개를 예측하기때문에 MANY-TO-ONE형태이다
        Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=None)
        return Y_pred
    
#def discriminator(x, nOutput=1, nHidden=nDHidden, nLayer=1, reuse=False):
#    with tf.variable_scope("discriminator", reuse=reuse):
#        h = slim.stack(x, slim.fully_connected, [nHidden] * nLayer, activation_fn=tf.nn.relu)
#        d = slim.fully_connected(h, nOutput, activation_fn=None)
#    return d    
##   
    
def discriminator(X, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        W1 = tf.get_variable("W1", shape=[1, 128],
                     initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.Variable(tf.random_normal([128]))
        L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
        L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
        
        W2 = tf.get_variable("W2", shape=[128, 64],
                             initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.Variable(tf.random_normal([64]))
        L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
        L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
        
#        W3 = tf.get_variable("W3", shape=[1000, 1000],
#                     initializer=tf.contrib.layers.xavier_initializer())
#        b3 = tf.Variable(tf.random_normal([1000]))
#        L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
#        L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
#        
#        W4 = tf.get_variable("W4", shape=[1000, 1000],
#                     initializer=tf.contrib.layers.xavier_initializer())
#        b4 = tf.Variable(tf.random_normal([1000]))
#        L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
#        L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
#        
#        W5 = tf.get_variable("W5", shape=[1000, 1000],
#                     initializer=tf.contrib.layers.xavier_initializer())
#        b5 = tf.Variable(tf.random_normal([1000]))
#        L5 = tf.nn.relu(tf.matmul(L4, W5) + b5)
#        L5 = tf.nn.dropout(L5, keep_prob=keep_prob)
#        
#        W6 = tf.get_variable("W6", shape=[1000, 1000],
#                     initializer=tf.contrib.layers.xavier_initializer())
#        b6 = tf.Variable(tf.random_normal([1000]))
#        L6 = tf.nn.relu(tf.matmul(L5, W6) + b6)
#        L6 = tf.nn.dropout(L6, keep_prob=keep_prob)
#        
#        W7 = tf.get_variable("W7", shape=[1000, 1000],
#                     initializer=tf.contrib.layers.xavier_initializer())
#        b7 = tf.Variable(tf.random_normal([1000]))
#        L7 = tf.nn.relu(tf.matmul(L6, W7) + b7)
#        L7 = tf.nn.dropout(L7, keep_prob=keep_prob)
#   
#        W8 = tf.get_variable("W8", shape=[1000, 1000],
#                     initializer=tf.contrib.layers.xavier_initializer())
#        b8 = tf.Variable(tf.random_normal([1000]))
#        L8 = tf.nn.relu(tf.matmul(L7, W8) + b8)
#        L8 = tf.nn.dropout(L8, keep_prob=keep_prob)
#        
#        W9 = tf.get_variable("W9", shape=[1000, 1000],
#                     initializer=tf.contrib.layers.xavier_initializer())
#        b9 = tf.Variable(tf.random_normal([1000]))
#        L9 = tf.nn.relu(tf.matmul(L8, W9) + b9)
#        L9 = tf.nn.dropout(L9, keep_prob=keep_prob)
#                
        W10 = tf.get_variable("W10", shape=[64, 1],
                             initializer=tf.contrib.layers.xavier_initializer())
        b10 = tf.Variable(tf.random_normal([1]))
        hypothesis = tf.sigmoid(tf.matmul(L2, W10) + b10)
#        hypothesis = tf.matmul(L2, W10) + b10

    return hypothesis

learning_rate = 0.0004
num_stacked_layers = 5
num_units = 120
keep_prob = 0.8
input_data_column_num = 19
iterations = 20    

batch_size = 10    

tf.reset_default_graph()
   
X = tf.placeholder(tf.float32, [1,None,input_data_column_num])
print(X)
Y = tf.placeholder(tf.float32, [None, 1])
print(Y)



#%%
gen_sample = generator(X)
# Build 2 Discriminator Networks (one from noise input, one from generated samples)
disc_real = discriminator(Y)
disc_fake = discriminator(gen_sample, reuse=True)
disc_concat = tf.concat([disc_real, disc_fake], axis=0)


disc_target = tf.placeholder(tf.float32, shape=[None,1])
print(disc_target)

gen_target = tf.placeholder(tf.float32, shape=[None,1])
print(gen_target)

stacked_gan = discriminator(gen_sample, reuse=True)


#%%

#
#disc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
#    logits=disc_concat, labels=disc_target))

#disc_loss=-tf.reduce_mean(disc_target* tf.log(disc_concat) + (1 - disc_target) *
#                       tf.log(1 - disc_concat))
disc_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=tf.ones_like(disc_real)) +
        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.zeros_like(disc_fake)))


#gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
#    logits=stacked_gan, labels=gen_target))
#
#gen_loss = -tf.reduce_mean(gen_target * tf.log(stacked_gan) + (1 - gen_target) *
#                       tf.log(1 - stacked_gan))

gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, 
                                                                  labels=tf.ones_like(disc_fake))) \
                                                                  + tf.reduce_mean(tf.square(gen_sample - Y)) 


optimizer_gen =  tf.train.AdamOptimizer(learning_rate=0.0004)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=0.00002)

gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
# Discriminator Network Variables
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')


train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


#%%

with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for i in range(iterations):
        for j in range(0,len(trainX)-1, batch_size):
            # Prepare Input Data
            # Get the next batch of MNIST data (only images are needed, not labels)
            batch_x = np.reshape(trainY[int(j//batch_size)], (1,1))
    
            # Generate noise to feed to the generator
            z = np.reshape(trainX[j:j+batch_size] , (1,10,19))
    
            # Prepare Targets (Real image: 1, Fake image: 0)
            # The first half of data fed to the generator are real images,
            # the other half are fake images (coming from the generator).
            batch_disc_y = np.concatenate(
                [np.ones([1]), np.zeros([1])], axis=0)
            batch_disc_y = np.reshape(batch_disc_y, (2,1))
            # Generator tries to fool the discriminator, thus targets are 1.
            batch_gen_y = np.ones([1])
            batch_gen_y = np.reshape(batch_gen_y, (1,1))
    
            # Training
            feed_dict = {Y: batch_x, X: z,
                         disc_target: batch_disc_y, gen_target: batch_gen_y}
            _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
                                    feed_dict=feed_dict)
            print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))
                
    test_prediction = []
    for i in range(0,len(testX)-1, batch_size):
        # Noise input.
        z = np.reshape(testX[i:i+batch_size], (1,10,19))
        g = sess.run(gen_sample, feed_dict={X: z})
        test_prediction.append(g)
#        for j in range(4):
#            # Generate image from noise. Extend to 3 channels for matplot figure.
#            img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
#                             newshape=(28, 28, 3))
#            a[j][i].imshow(img)

#    f.show()
#    plt.draw()
#    plt.waitforbuttonpress()
        
#%%
prediction = np.reshape(test_prediction, (len(test_prediction),1))
#prediction= np.reshape(test_prediction, (837,1))
flattenY = np.reshape(testY, (len(testY),1))
#pd.DataFrame(prediction).to_csv('56%1.csv')
#del test_predict, testY

plt.plot(prediction)
plt.plot(flattenY)

    
#%%
a = pd.DataFrame(flattenY) - pd.DataFrame(flattenY).shift(1)
b = pd.DataFrame(prediction) - pd.DataFrame(flattenY).shift(1)  
#accuracy = ( sum( (a>0).values * (b>0).values ) + sum( (a<0).values * (b<0).values) )/len(flattenY)
accuracy = sum( (a.values * b.values)>0 )/len(flattenY)
print(accuracy)


#%%
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    













    
    
    

























    
    
    
    
    
    