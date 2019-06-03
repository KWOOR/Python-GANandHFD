# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 15:18:15 2019

@author: rbgud
"""
import talib
from talib.abstract import Function
import pandas as pd

#%%
def get_func_list():
    
    talib_func = talib.get_function_groups()
    
    return talib_func

#%%
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

#%%
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



