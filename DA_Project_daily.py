# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:21:38 2019

@author: kur7
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 11:04:22 2019

@author: 우람
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

data = df_final[name[0]] #5


ticker=data.columns[2:-1]
pred_days=1 #몇일 동안의 데이터로 다음날것을 예측할건지.. 
x = np.zeros((len(data)-pred_days,len(ticker) ))
y = np.zeros((len(data)-pred_days,1 ))

for i in range(0,len(data)-pred_days):
    x[i]=data.iloc[i:i+pred_days][ticker].values
    y[i]=data.iloc[i+pred_days]['Price']
  
#pred_days = 20  
#trainX = []
#trainY = []
#ticker=data.columns[2:-1]
#
#for i in range(0,len(data)-pred_days):
#    trainX.append(data.iloc[i:i+pred_days][ticker].values)
#    trainY.append(data.iloc[i+pred_days]['Price'])    
#    
    

del data
#%%
def gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * math.pow(x, 3))))

rolling_window = 5 #5일씩 롤링
train_period = 20 #20일 동안 학습하고
test_period = 5 #5일동안 테스트

seq_length = 20
input_data_column_num = 19
learning_rate = 0.0004
num_stacked_layers = 5
num_units = 120
keep_prob = 0.8
iterations = 20
trainX = []
testX = []
trainY = []
testY = []



for i in np.arange(0,len(x)-train_period-test_period,rolling_window):
    
    trainX.append(x[i:i+train_period])
    
    trainY.append(y[i:i+train_period])
    
    testX.append(x[i+train_period:i+train_period+test_period])

    testY.append(y[i+train_period:i+train_period+test_period])
    
del x,y


#%%

''' 트레인 셋 따로, 테스트 셋 따로 민맥스 스케일링을 한다! PCA에서는 스탠다드가 좋고 일반 뉴럴넷에서는 민맥스가 좋음'''
''' 따라서.. 뉴럴넷을 돌리기 위해 민맥스 스케일링을 하는것임! '''
#
#trainX = MinMaxScaler(trainX)
#trainY = MinMaxScaler(trainY)
#testX = MinMaxScaler(testX)
#testY = MinMaxScaler(testY)
#
#def refine_data(data):
#    buff = pd.DataFrame()
#    for i in range(len(data)):
#        buff = pd.concat( [buff, pd.DataFrame(data[i])], axis=0)
#    return MinMaxScaler(buff).values

def refine_data(data):
    buff = pd.DataFrame()
    for i in range(len(data)):
        buff = pd.concat( [buff, pd.DataFrame(data[i])], axis=0)
    return buff.values

trainX = refine_data(trainX)
trainY = refine_data(trainY)
testX = refine_data(testX)
testY = refine_data(testY)



#%%
tf.reset_default_graph()

# 텐서플로우 플레이스홀더 생성
X = tf.placeholder(tf.float32, [1, None, input_data_column_num])
print("X: ", X)
Y = tf.placeholder(tf.float32, [None, 1])
print("Y: ", Y)

# 검증용 측정지표를 산출하기 위한 targets, predictions를 생성한다
targets = tf.placeholder(tf.float32, [None, 1])
print("targets: ", targets)

predictions = tf.placeholder(tf.float32, [None, 1])
print("predictions: ", predictions) 

#%%   

def LSTM_cell(keep_prob):
    # forget_bias: biases of the forget gate (default: 1) in order to reduce the scale of forgetting in the beginning of the training.
    
    cell = tf.contrib.rnn.LSTMCell(num_units= num_units,state_is_tuple=True,
                                   initializer= tf.contrib.layers.xavier_initializer())
    if keep_prob < 1.0:
         cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob, state_keep_prob= keep_prob)
    return cell

# num_stacked_layers개의 층으로 쌓인 Stacked RNNs 생성
stackedRNNs = [LSTM_cell(keep_prob) for _ in range(num_stacked_layers)]
multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs, state_is_tuple=True) 

# RNN Cell(여기서는 LSTM셀임)들을 연결
outputs, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)
print("outputs: ", outputs)

# [:, -1]를 잘 살펴보자. LSTM RNN의 마지막 (hidden)출력만을 사용했다.
# 과거 여러 거래일의 주가를 이용해서 다음날의 주가 1개를 예측하기때문에 MANY-TO-ONE형태이다
Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=None)

#%%
# Loss function define

loss_p = tf.reduce_sum(tf.square(Y_pred - Y)) 
#loss_dpl = tf.reduce_sum(tf.abs(tf.sign(Y_pred[1:,:] - Y[:-1,:]) - tf.sign(Y[1:,:] - Y[:-1,:])  ) )
#loss_dpl = tf.reduce_sum(Y_pred - Y)

loss = loss_p

# Optimizer
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# RMSE(Root Mean Square Error)
rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(targets, predictions)))

#%%

# Session 초기화
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # 모델 학습
    prev_training_loss = 0.0
    for i in range(iterations):
        for j in range(len(trainX)):
            _, step_loss = sess.run([train, loss], feed_dict={
                                    X:np.reshape(trainX[j], (1,1 ,input_data_column_num)) , 
                                    Y: np.reshape(trainY[j], (1,1) )})
            print("[step: i:{}, j:{}] loss: {}".format(i,j, step_loss))
            
            # 이전 loss와 현재 loss의 차이가 유의미하게 적다고 판단되면 학습을 조기 종료 시킨다.
#            if np.abs(prev_training_loss - step_loss) < 0.000000001:
#                print ("early stopping..")
#                break
            prev_training_loss = step_loss

    # 모델 테스트
    test_predict = []
    for i in range(len(testX)):
        test_predict.append(sess.run(Y_pred, feed_dict={X: np.reshape(testX[i], 
                                                                      (1,1,input_data_column_num))}))
    
# 결과 값 출력
del trainX, trainY, testX

#%%
prediction = np.reshape(test_predict, (len(test_predict),1))
flattenY = np.reshape(testY, (len(testY),1))
pd.DataFrame(prediction).to_csv('56%.csv')
del test_predict, testY

plt.plot(prediction)
plt.plot(flattenY)

    
#%%

#BRK = pd.read_csv('BRB[5].csv').iloc[:,1]
#plt.plot((BRK+1).cumprod())

#prediction = pd.read_csv('63%_RMSE7059.csv').iloc[:,1]

a = pd.DataFrame(flattenY) - pd.DataFrame(flattenY).shift(1)
b = pd.DataFrame(prediction) - pd.DataFrame(flattenY).shift(1)  
#accuracy = ( sum( (a>0).values * (b>0).values ) + sum( (a<0).values * (b<0).values) )/len(flattenY)
accuracy = sum( (a.values * b.values)>0 )/len(flattenY)
print(accuracy)

#%%


print(sum(sum(np.array(prediction>=0).reshape(-1,len(prediction)) == (flattenY>=0).reshape(-1,len(flattenY))))/len(flattenY))

print(np.sqrt(sum(sum( ( (np.array(prediction).reshape(-1,len(prediction)) - flattenY.reshape(-1,len(flattenY)))/(flattenY.reshape(-1,len(flattenY))+1e-7) ) **2)) / len(flattenY)))
    



print(sum(sum(np.array(prediction).reshape(-1,len(prediction)) == (flattenY).reshape(-1,len(flattenY))))/len(flattenY))

#%%
EMA = 0.0
gamma = 0.1
buff = np.zeros_like(prediction)
for ti in range(len(prediction)):
  EMA = gamma*prediction[ti] + (1-gamma)*EMA
  buff[ti] = EMA
  
plt.plot(buff)
plt.plot(flattenY)


a = pd.DataFrame(flattenY) - pd.DataFrame(flattenY).shift(1)
b = pd.DataFrame(buff) - pd.DataFrame(flattenY).shift(1)  
#accuracy = ( sum( (a>0).values * (b>0).values ) + sum( (a<0).values * (b<0).values) )/len(flattenY)
accuracy = sum( (a.values * b.values)>0 )/len(flattenY)
print(accuracy)

pd.DataFrame(buff).to_csv('trend_success.csv')

#%%

































