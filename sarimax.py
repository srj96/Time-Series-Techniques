import pandas as pd 
import numpy  as np
import statsmodels.api as sm 
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf

class SarimaModel:

    '''Below calculated values : Return Series , PACF and ACF , Arima Prediction , Transformation '''

    def __init__(self,**kwargs):
        attr_list = set(['train_path','train_data','test_data','train_rs','test_rs',
                         'test_path','rlags','acf_lags','pacf_lags','acf_max_df',
                         'pacf_max_df','sarima_data','file_path'])
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

    # Return Series with shift given in the main() < presently calculated for t-1>
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
    
    # PACF and ACF with number of lags given in main() < presently given lags = 30>
    def lag_cal(self):
        # self.train_rs['lag_1'] = self.train_rs['lag_1'].dropna()
        # train_rs_cf = self.train_rs
        train_rs_cf = self.train_rs[self.train_rs['lag_1'] != np.inf]
        self.acf_max_df = pd.DataFrame()
        for ix in self.train_rs['period'].unique():
            acf_val = acf(train_rs_cf[train_rs_cf['period'] == ix]['lag_1'], nlags= self.acf_lags, fft = False)
            acf_max_val = max(abs(acf_val[acf_val != 1.0]))
            acf_list = list(abs(acf_val))
            acf_max_list = acf_list.index(acf_max_val)
            p = {'period': ix , 'lag_acf' : acf_max_list}
            df_1 = pd.DataFrame(data = p, index = [0])
            self.acf_max_df = self.acf_max_df.append(df_1)
    
    
        self.pacf_max_df = pd.DataFrame()
        for iy in self.train_rs['period'].unique():
            pacf_val = pacf(train_rs_cf[train_rs_cf['period'] == iy]['lag_1'], nlags= self.pacf_lags)
            pacf_max_val = max(abs(pacf_val[pacf_val != 1.0]))
            pacf_list = list(abs(pacf_val))
            pacf_max_list = pacf_list.index(pacf_max_val)
            p = {'period' : iy , 'lag_pcf' : pacf_max_list}
            df_2 = pd.DataFrame(data = p, index = [0]) 
            self.pacf_max_df = self.pacf_max_df.append(df_2)
        
        return(list(self.acf_max_df['lag_acf']),list(self.pacf_max_df['lag_pcf']))
    
    # Arima Model predictions with values of p and q picked from lag_cal() 
    def sr_model(self):
        train_rs_ar = self.train_rs[self.train_rs['lag_1'] != np.inf]
        self.sarima_data = pd.DataFrame()
        for ix in sorted(self.train_rs['period'].unique()):
            train = train_rs_ar[train_rs_ar['period'] == ix]['lag_1']
            test = self.test_rs[self.test_rs['period'] == ix]['lag_1']
            lag_q = [q for iq , q in enumerate(self.acf_max_df[self.acf_max_df['period'] == ix]['lag_acf'])][0]
            lag_p = [p for ip , p in enumerate(self.pacf_max_df[self.pacf_max_df['period']== ix]['lag_pcf'])][0]
    
            model = sm.tsa.statespace.SARIMAX(train,trend = 'n', order = (lag_p,0,lag_q), seasonal_order=(1,0,0,7))
            srm = model.fit(disp = 0)
            output = srm.forecast(steps = len(test))
            yhat = output.values
            print('time_slot',ix)
            p = {'date': self.test_rs[self.test_rs['period'] == ix]['date'],'period': ix , 'pred_val': yhat}
            df_1 = pd.DataFrame(data = p)
            self.sarima_data = self.sarima_data.append(df_1)
        return(self.sarima_data)
    
    # Exponentiation of the log transformed values giving price prediction 
    def sr_prediction(self):
        self.sarima_data = self.sarima_data.reset_index()
        self.sarima_data = self.sarima_data[['date','period','pred_val']]

        sarima_data_full = pd.merge(self.sarima_data,self.test_data , on=['date','period'])
        pred_price = np.exp(sarima_data_full['pred_val'])*(sarima_data_full['price'].shift(periods = 1 , fill_value = 0))
        sarima_data_full['predicted_price'] = pred_price
        sarima_data_pred = sarima_data_full[sarima_data_full['date'] != '2019-07-31']
        error = abs(sarima_data_pred['price'] - sarima_data_pred['predicted_price'])
        ape = ((error)/sarima_data_pred['price'])*100 
        sarima_data_pred['%_Error'] = ape
        sarima_data_pred = sarima_data_pred.rename(columns = {'price' : 'actual_price'})
        sarima_data_pred = sarima_data_pred.round({'%_Error' : 2})
        
        with open(self.file_path,'w') as savefile:
            sarima_data_pred.to_csv(savefile)
        
        print(sarima_data_pred)
        return(sarima_data_pred)
      

if __name__ == '__main__':
    
    train_path = '/home/congnitensor/Python_projects/ptc_armax/train_test_ptc/train_ptc_n1.csv'
    test_path = '/home/congnitensor/Python_projects/ptc_armax/train_test_ptc/test_ptc_n1.csv'
    file_path = '/home/congnitensor/Python_projects/ptc_armax/ptc_pred_sarima/ptc_area_n1_pred.csv'
    rlags = 1
    acf_lags = 30
    pacf_lags = 30
    obj = SarimaModel(train_path = train_path, test_path = test_path , rlags = rlags, 
                      acf_lags = acf_lags, pacf_lags = pacf_lags, train_data = None, 
                      test_data = None, train_rs = None, test_rs = None, acf_max_df = None, 
                      pacf_max_df = None,sarima_data = None, file_path = file_path)
                    
    print(obj.read_data())
    print(obj.return_series())
    print(obj.lag_cal())
    obj.sr_model()
    obj.sr_prediction()



# model = sm.tsa.statespace.SARIMAX(train.price, trend='n', order=(1,1,7), seasonal_order=(1,0,0,7)).fit()
# output1 = model.forecast(steps = len(test2))
# print(output1)