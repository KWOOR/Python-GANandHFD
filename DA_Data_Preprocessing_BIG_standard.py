# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:20:40 2019

@author: kur7
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 18:54:53 2019

@author: 우람
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from talib.abstract import *
import talib
from talib.abstract import Function
import pandas as pd
import seaborn as sa
import sys
from sklearn.decomposition import PCA
from sklearn import preprocessing
import statsmodels.api as sm
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os
import sys
mod = sys.modules[__name__]
os.chdir('C:\\Users\\kur7\\OneDrive\\바탕 화면\\uiuc\\DA')

#%%
def get_func_list():
    
    talib_func = talib.get_function_groups()
    
    return talib_func

def get_talib_func(Indicator_Groups,inputs):
    
    ta_dict = get_func_list()
    timeperiod = [5,10,30]
    mom_timeperiod = [5,30]
    empty = pd.DataFrame()
     
    for j in range(len(ta_dict[Indicator_Groups])):
        
        if Indicator_Groups == 'Overlap Studies':
           if ta_dict[Indicator_Groups][j] in ['HT_TRENDLINE','MAMA','KAMA','MAVP',\
                     'MA','SAR','SAREXT','MIDPOINT','MIDPRICE','TRIMA']:
              pass 
             
           elif ta_dict[Indicator_Groups][j] =='BBANDS':
                direction = ['up','mid','low']
                for k in range(3):
                     empty['{0}_{1}'.format(ta_dict[Indicator_Groups][j].lower(),direction[k])] = \
                          Function(ta_dict[Indicator_Groups][j])(inputs)[k]
                     
           else:    
                for time in timeperiod:

                    empty['{0}_{1}'.format(ta_dict[Indicator_Groups][j].lower(),time)] = \
                         Function(ta_dict[Indicator_Groups][j])(inputs,time)
                    
                            
        if Indicator_Groups == 'Volatility Indicators':    
           if ta_dict[Indicator_Groups][j] == 'TRANGE':
              empty['{0}'.format(ta_dict[Indicator_Groups][j].lower())] = \
                     Function(ta_dict[Indicator_Groups][j])(inputs)
           else:
              for time in timeperiod:
                  empty['{0}_{1}'.format(ta_dict[Indicator_Groups][j].lower(),time)] = \
                         Function(ta_dict[Indicator_Groups][j])(inputs,time) 
            
        if Indicator_Groups == 'Volume Indicators':                 
           empty['{0}'.format(ta_dict[Indicator_Groups][j].lower())] = \
                     Function(ta_dict[Indicator_Groups][j])(inputs)
                         
        if Indicator_Groups == 'Momentum Indicators':
           if ta_dict[Indicator_Groups][j] in ['ADX','ADXR','AROON','MACD','MACDEXT','MACDFIX',\
                     'MINUS_DI','MINUS_DM','PLUS_DI','PLUS_DM','ROC','CMO','WILLR','MOM','ROCP','ROCR','BOP']:
              pass
           elif ta_dict[Indicator_Groups][j] == 'STOCH':
                empty['slowk'] = Function(ta_dict[Indicator_Groups][j])(inputs)[0]
                empty['slowd'] = Function(ta_dict[Indicator_Groups][j])(inputs)[1]
                
           elif ta_dict[Indicator_Groups][j] == 'STOCHF':     
                empty['fastk'] = Function(ta_dict[Indicator_Groups][j])(inputs)[0]
                empty['fastd'] = Function(ta_dict[Indicator_Groups][j])(inputs)[1]
           
           elif ta_dict[Indicator_Groups][j] == 'STOCHRSI':
                empty['strsi_fastk'] = Function(ta_dict[Indicator_Groups][j])(inputs)[0]
                empty['strsi_fastd'] = Function(ta_dict[Indicator_Groups][j])(inputs)[1]
                
           elif ta_dict[Indicator_Groups][j] == 'ULTOSC':    
                empty['ultosc{0}'.format(51020)] = Function(ta_dict[Indicator_Groups][j])(inputs,5,10,20)
                empty['ultosc{0}'.format(3612)] = Function(ta_dict[Indicator_Groups][j])(inputs,3,6,12)
        
           
           elif ta_dict[Indicator_Groups][j] in ['APO','PPO']: 
                empty['apo'] = Function('APO')(inputs)
#                empty['bop'] = Function('BOP')(inputs)
                empty['ppo'] = Function('PPO')(inputs)
               
           else:
               for time in mom_timeperiod:
                  empty['{0}_{1}'.format(ta_dict[Indicator_Groups][j].lower(),time)] = \
                         Function(ta_dict[Indicator_Groups][j])(inputs,time) 
                
    
    return empty

def generate_ta(Indicator_Groups,inputs):   
    
    ov_list = []
    vol_list = []
    volume_list = []
    mom_list = []
    ta_list = []
    for i in range(len(inputs)):
        
        ov = get_talib_func(Indicator_Groups[0],inputs[i])
        ov_list.append(ov)
        vol = get_talib_func(Indicator_Groups[1],inputs[i])
        vol_list.append(vol) 
        volume = get_talib_func(Indicator_Groups[2],inputs[i])
        volume_list.append(volume) 
        mom = get_talib_func(Indicator_Groups[3],inputs[i])
        mom_list.append(mom)
        
    for j in range(len(inputs)):
        ta_df = pd.concat([ov_list[j],vol_list[j],volume_list[j],mom_list[j]],axis=1)
        ta_list.append(ta_df)
    return ta_list    



def make_name(data):
    name=dict(zip(range(len(data)),data.keys()))
    name_buff=dict(zip(data.keys(),range(len(data))))
    return name, name_buff

#%%

df1 = pd.read_csv('Final_data1.csv').iloc[:,1:]
df2 = pd.read_csv('Final_data2.csv').iloc[:,1:]
data = pd.concat([df1,df2])
df=dict(list(data.groupby(data['Ticker'])))
del df1, df2, data
name,_ = make_name(df)

input_list = []
for i in range (len(df)):
    
    inputs = {
    'open': df[name[i]]['Open'],
    'high': df[name[i]]['High'],
    'low': df[name[i]]['Low'],
    'close': df[name[i]]['Close'],
    'volume': np.asarray(df[name[i]]['Volume'], dtype='float')}
   
    input_list.append(inputs)
del inputs

#%%
    
#print(tc.get_func_list())    

Indicator_Groups = ['Overlap Studies','Volatility Indicators','Volume Indicators','Momentum Indicators']      

ta_list = generate_ta(Indicator_Groups,input_list)

df_new = []
for i in range(len(df)):
    
    column_name = df[list(df.keys())[i]].columns.tolist()+ta_list[i].columns.tolist()
    new_df = pd.concat([df[list(df.keys())[i]].reset_index(drop=True),ta_list[i].astype(float)],ignore_index=True,axis=1)
    new_df.columns = column_name
    df_new.append(new_df)

del new_df



#%%
""" Code to create the Fuorier trasfrom  """

def FT_feature(data):
    ft_list = []
    for i in range(len(data)):
        
        data_FT = data[i][['Dates', 'Close']]
        close_fft = np.fft.fft(np.asarray(data_FT['Close'].tolist()))
        fft_df = pd.DataFrame({'fft':close_fft})
        fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
        fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
        fft_list = np.asarray(fft_df['fft'].tolist())
        for num_ in [3, 6, 9]:
            fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_]=0
            ft= pd.DataFrame(np.fft.ifft(fft_list_m10))
            ft_list.append(ft)
 
    return ft_list
    

ft_features = FT_feature(df_new)



#%%
df_here = df_new
finaldata= []
for idx, value in enumerate(ft_features):
    
    if idx%3 == 0:
        
        df_plus= pd.DataFrame()
        a = pd.DataFrame(df_here[int(idx/3)])
        b = pd.DataFrame(ft_features[idx].values.real,columns =['ft3'])
        c = pd.DataFrame(ft_features[idx+1].values.real,columns =['ft6'])
        d = pd.DataFrame(ft_features[idx+2].values.real,columns =['ft9'])
        df_plus = pd.concat([a,b], axis =1) 
        df_plus = pd.concat([df_plus,c], axis =1) 
        df_plus = pd.concat([df_plus,d], axis =1) 
        
        finaldata.append(df_plus)
        continue
    
    elif idx%3 == 1:
        continue
    
    elif idx%3 ==2:
        continue
    
    elif idx == (len(ft_features)-3):
        break 
    
for i in range(len(finaldata)):
    finaldata[i]['return']=pd.DataFrame(finaldata[i]['Close']/finaldata[i]['Close'].shift(1)-1)
    
del df_here, df_new
del d, df, df_plus, ft_features
del a,b,c
del ta_list, value
del input_list, column_name, Indicator_Groups

#%%
''' 빅데이터 만들기 '''

num = len(finaldata[0].columns)-2 #앞에 2개는 스트링이라 제외...
for i in range(len(finaldata)):
    finaldata[i] = finaldata[i].dropna()
    for j in range(2,num):
        finaldata[i] = pd.concat( [finaldata[i], 
                 pd.DataFrame(finaldata[i].iloc[:,j]*finaldata[i].iloc[:,j])], axis=1 )


#%%          
for i in range(len(finaldata)):
    finaldata[i] = finaldata[i].dropna()
    for j in range(2,num-1):
        finaldata[i] = pd.concat( [finaldata[i], 
                 pd.DataFrame(finaldata[i].iloc[:,j]*finaldata[i].iloc[:,j+1])], axis=1 )
               
               
#%%
for i in range(len(finaldata)):
    finaldata[i] = finaldata[i].dropna()
    for j in range(2,num-2):
        finaldata[i] = pd.concat( [finaldata[i], 
                 pd.DataFrame(finaldata[i].iloc[:,j]*finaldata[i].iloc[:,j+2])], axis=1 )



#%%
# PCA를 위해서는 Standardization(?)을 한다! 

norm_finaldata = []
for i in range(len(finaldata)):
    a0 = finaldata[i].loc[390:,:'Ticker']
    a1 = pd.DataFrame(preprocessing.StandardScaler().fit_transform(finaldata[i].iloc[len(finaldata[i])-len(a0):,2:]))
    a2 = pd.concat([pd.DataFrame(a0.values),pd.DataFrame(a1.values)], axis=1 ,  sort=False)
    norm_finaldata.append(a2)   

del a0, a1, a2

total = pd.DataFrame()
for i in tqdm(range(len(norm_finaldata))):
    
    con = norm_finaldata[i]
    total = pd.concat([total,con],ignore_index=True)         

del con,norm_finaldata, finaldata
#%%

totalX = total.iloc[:,2:] # np.array(total.loc[:,'Volume':])[:1000,:]  


pca = PCA()
X=pca.fit_transform(totalX.values)
#Y=pca.fit_transform(df_new[0]['Close'].values.reshape(-1,1))

np.array(sorted(pca.explained_variance_, reverse=True))[:19].sum()/np.array(sorted(pca.explained_variance_, reverse=True))[:].sum()

totalY = total.loc[:,'Close']#np.array(total['Close'])[:1000] 

#%%
#def lasso_picking_feature(data):
#    buff = data.dropna()
#    index = list(buff.loc[:,'Open':'ft9'].columns)
#    X= buff[index].values
##    ones = np.ones(len(X))
##    X = sm.add_constant(np.column_stack((X, ones)))
#    Y = buff['return'].values
#    model = sm.OLS(Y, X)
#    result = model.fit_regularized(alpha=0.00001, L1_wt=1)
##    index.append('const')
#    buff = pd.concat([pd.DataFrame(index, columns=['index']),pd.DataFrame(result.params, columns = ['params'])], axis=1)
#    features = buff[buff['params']!=0]['index'].values
#    return features
#
#
#df_new0_feature = lasso_picking_feature(total)
#print(df_new0_feature)
#
#
#
##%%
#X_train, X_test, y_train, y_test = train_test_split(totalX,totalY, test_size=0.33)
#regressor = xgb.XGBRegressor(gamma=0.0,n_estimators=150,base_score=0.7,colsample_bytree=1,learning_rate=0.05)
# 
#del totalX, totalY
#xgbModel = regressor.fit(X_train,y_train, 
#                         eval_set = [(X_train, y_train), (X_test, y_test)], verbose=False)
#fig = plt.figure(figsize=(8,8))
#plt.xticks(rotation='vertical')
#plt.bar([i for i in range(len(xgbModel.feature_importances_))], 
#                         xgbModel.feature_importances_.tolist(), tick_label=X_test.columns)
#plt.title('Figure 6: Feature importance of the technical indicators.')
#plt.show()     
#
#del X_train, X_test, y_train, y_test
#
#
##%%
#total_final = total.loc[:,['Dates','Ticker','cci_5','cci_30','rsi_5','rsi_30','slowk','fastk',\
#                           'strsi_fastk','strsi_fastd','ultosc51020','ultosc3612']]
#
#pca_total = pd.DataFrame(X[:,:8],columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8'])
#
#real_final = pd.concat([total_final,pca_total,total['return']],axis=1)
#
#del total_final, pca_total
#del buff, data, X, Y
#del total, totalX, totalY
#
#real_final.iloc[:int(len(real_final)/2),:].to_csv('real_final_data1.csv')
#real_final.iloc[int(len(real_final)/2):,:].to_csv('real_final_data2.csv')


#%%
#total=total.dropna() #할 필요가 없지..
Big_total = pd.DataFrame(total.iloc[:,:2].values, columns=['Dates', 'Ticker'])
Big_pca_total = pd.DataFrame(X[:,:19],columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9',
                             'PC10','PC11', 'PC12','PC13','PC14','PC15','PC16','PC17','PC18','PC19'])
Big_Y = pd.DataFrame(total.iloc[:,3].values, columns=['Close']) #Close의 인덱스임! 
Big_final = pd.concat([Big_total, Big_pca_total, Big_Y], axis=1)
    
Big_final.iloc[:int(len(Big_final)/2),:].to_csv('Big_std_final_data1.csv')
Big_final.iloc[int(len(Big_final)/2):,:].to_csv('Big_std_final_data2.csv')

''' 민맥스 스케일링이나 스무딩은 안 되어있음! 오직 스탠다이징하고 피씨에이만 해서 피쳐만 추림 '''  
    