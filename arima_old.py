import pandas as pd 
import numpy  as np
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf

class ArimaModel:

    def __init__(self,**kwargs):
        attr_list = set(['train_path','train_data','test_data','train_rs','test_rs',
                         'test_path','lag_p','lag_q','rlags','acf_lags'])
        [self.__setattr__(k,kwargs[k]) for k in kwargs.keys() if k in attr_list ]
    
    def read_data(self):
        with open(self.train_path, 'r') as train_file:
            self.train_data = pd.read_csv(train_file)
        with open(self.test_path, 'r') as test_file:
            self.test_data = pd.read_csv(test_file)
        
        self.train_data = self.train_data.reset_index()
        self.train_data = self.train_data[['date', 'period', 'price']]

        self.test_data = self.test_data.reset_index()
        self.test_data = self.test_data[['date', 'period' , 'price']]

        return(self.train_data,self.test_data)

    def return_series(self):
        self.train_rs = pd.DataFrame()
        for ix in self.train_data['period'].unique():
            rs_val = np.log(self.train_data[self.train_data['period'] == ix]['price'] / self.train_data[self.train_data['period'] == ix]['price'].shift(periods = self.rlags, fill_value = 0))
            p = {'date' : self.train_data[self.train_data['period'] == ix]['date'], 'period' : ix , 'lag_{}'.format(self.rlags) : rs_val}
            df = pd.DataFrame(data = p)
            self.train_rs = self.train_rs.append(df)
        
        self.test_rs = pd.DataFrame()
        for iy in self.test_data['period'].unique():
            rs_val_new = np.log(self.test_data[self.test_data['period'] == iy]['price'] / self.test_data[self.test_data['period'] == iy]['price'].shift(periods = self.rlags, fill_value = 0))
            p = {'date' : self.test_data[self.test_data['period'] == iy]['date'], 'period' : iy , 'lag_{}'.format(self.rlags) : rs_val_new}
            df_new = pd.DataFrame(data = p)
            self.test_rs = self.test_rs.append(df_new)

        return(self.train_rs,self.test_rs) 
    
    def lag_cal(self):
        train_rs_cf = self.train_rs[self.train_rs['lag_1'] != np.inf]
        for ix in self.train_rs['period'].unique():
            acf_val = acf(train_rs_cf[train_rs_cf['period'] == ix]['lag_1'], nlags= self.acf_lags, fft = False)
            acf_max_val = max(abs(acf_val[acf_val != 1.0]))
            acf_list = list(abs(acf_val))
            acf_max_pos = acf_list.index(acf_max_val)
            ser = pd.Series(acf_max_pos)
            lst = ser.tolist()
            print(lst)
            
        

        # for iy in self.train_rs['period'].unique():
        #     pacf_val = pacf(train_rs_cf[train_rs_cf['period'] == iy]['lag_1'], nlags= 30)
        #     pacf_max_val = max(abs(pacf_val[pacf_val != 1.0]))
        #     pacf_list = list(abs(pacf_val))
        #     pacf_max_pos = pacf_list.index(pacf_max_val) 






    

if __name__ == '__main__':
    
    train_path = 'C:/Users/SHIVAM RAWAT/Pictures/Python_Programs/train_ptc.csv'
    test_path = 'C:/Users/SHIVAM RAWAT/Pictures/Python_Programs/test_ptc.csv'
    rlags = 1
    acf_lags = 30
    obj = ArimaModel(train_path = train_path, test_path = test_path , lag_p = None, lag_q = None, 
                      rlags = rlags, acf_lags = acf_lags, train_data = None, test_data = None, 
                      train_rs = None, test_rs = None)
                    
    obj.read_data()
    obj.return_series()
    obj.lag_cal() 





