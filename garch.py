from arch import arch_model 
import pandas as pd 
import numpy  as np
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf

class GarchModel:

    '''Below calculated values : Return Series , PACF and ACF , GARCH Prediction , Transformation '''

    def __init__(self,**kwargs):
        attr_list = set(['train_path','train_data','test_data','train_rs','test_rs',
                         'test_path','rlags','acf_lags','pacf_lags','acf_max_df',
                         'pacf_max_df','garch_data','file_path','arima_file_path',
                         'res_data'])
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
    
    # Garch Model predictions with values of p and q picked from lag_cal() 
    def gr_model(self):
        train_rs_ar = self.train_rs[self.train_rs['lag_1'] != np.inf]
        with open(arima_file_path,'r') as read_file:
            self.res_data = pd.read_csv(read_file)
        res_df = self.res_data[['period','%_Error']]
        self.garch_data = pd.DataFrame()
        for ix in sorted(self.train_rs['period'].unique()):
            train = train_rs_ar[train_rs_ar['period'] == ix]['lag_1']
            test = self.test_rs[self.test_rs['period'] == ix]['lag_1']
            lag_q = [q for iq , q in enumerate(self.acf_max_df[self.acf_max_df['period'] == ix]['lag_acf'])][0]
            lag_p = [p for ip , p in enumerate(self.pacf_max_df[self.pacf_max_df['period']== ix]['lag_pcf'])][0]
    
            garch_train = res_df[res_df['period'] == ix]['%_Error']
            model = arch_model(garch_train, mean = 'Zero', vol = 'GARCH', p = lag_p, q = lag_q)
            garch = model.fit()
            output = garch.forecast(horizon = len(test))
            # yhat = garch.params
            # res = [r for ir,r in enumerate(garch.resid)]
            # res = garch.resid.var()
            # vol = [vol_val for ivol_val, vol_val in enumerate(garch.conditional_volatility)]
            print('time_slot',ix)
            var = output.variance.values[-1,:]
            std_dev = np.sqrt(output.variance.values[-1,:])
            print(var)
            print(std_dev)
            print(garch_train) 
            # pred = output.forecasts
            # print(output.mean.dropna())
            # print(output.variance.dropna())
            # print(output.residual_variance.dropna())
            # orv = output.residual_variance.dropna()
            # ov = output.variance.dropna()
            # mean = output.mean.dropna()
            # res_var = [r for ir,r in enumerate(orv.values)][0]         
            # var = [v for iv, v in enumerate(ov.values)][0]

            # print(res_var)
            # print(orv)
            # print(var)

            # mu = [m for im,m in enumerate(mean.values)][0]
            # eps = [e for ie , e in enumerate(res_var.values)][0]

            # r = mu + (eps*res)
            # print(r)

            # sim = output.simulations     
            # pred_val = sim.values[-1,:,-1]
            # print(len(sim.values[-1,:,-1]))
            # print(yhat)
            # print(exp)
            # print(res)
            # print(len(res))
            # print(vol)
            # std_dev = np.sqrt(yhat['omega'] + yhat['alpha[1]']*res_var+ yhat['beta[1]']*var)
            # std_dev = np.sqrt(yhat['omega'] + yhat['alpha[1]']*res**2 + yhat['beta[1]']*vol**2)
            # print('standard_deviation : ', std_dev*res)
            # r = mu + (std_dev*res)
            p = {'date': self.test_rs[self.test_rs['period'] == ix]['date'],'period': ix , 'variance': var , 'std_deviation' : std_dev}
            df_1 = pd.DataFrame(data = p)
            self.garch_data = self.garch_data.append(df_1)
        return(self.garch_data)
    
    def hybrid_prediction(self):
        self.garch_data = self.garch_data.reset_index()
        self.garch_data = self.garch_data[['date','period','variance','std_deviation']]
        self.garch_data = self.garch_data[s]
    # Exponentiation of the log transformed values giving price prediction 
    def gr_prediction(self):
        self.garch_data = self.garch_data.reset_index()
        self.garch_data = self.garch_data[['date','period','pred_val']]

        garch_data_full = pd.merge(self.garch_data,self.test_data , on=['date','period'])
        pred_price = np.exp(garch_data_full['pred_val'])*(garch_data_full['price'].shift(periods = 1 , fill_value = 0))
        garch_data_full['predicted_price'] = pred_price
        garch_data_pred = garch_data_full[garch_data_full['date'] != '2019-07-31']
        error = abs(garch_data_pred['price'] - garch_data_pred['predicted_price'])
        ape = ((error)/garch_data_pred['price'])*100 
        garch_data_pred['%_Error'] = ape
        garch_data_pred = garch_data_pred.rename(columns = {'price' : 'actual_price'})
        garch_data_pred = garch_data_pred.round({'%_Error' : 2})
        garch_data_pred = garch_data_pred.round({'predicted_price' : 2})
        
        with open(self.file_path,'w') as savefile:
            garch_data_pred.to_csv(savefile)
        
        print(garch_data_pred)
        return(garch_data_pred)
      

if __name__ == '__main__':
    
    train_path = '/home/congnitensor/Python_projects/ptc_armax/train_test_ptc/train_ptc_e1.csv'
    test_path = '/home/congnitensor/Python_projects/ptc_armax/train_test_ptc/test_ptc_e1.csv'
    file_path = '/home/congnitensor/Python_projects/ptc_armax/ptc_pred_garch/ptc_garch_base_pred.csv'
    arima_file_path = '/home/congnitensor/Python_projects/ptc_armax/ptc_area_pred/ptc_area_e1_pred.csv'
    rlags = 1
    acf_lags = 30
    pacf_lags = 30
    obj = GarchModel(train_path = train_path, test_path = test_path , rlags = rlags, 
                      acf_lags = acf_lags, pacf_lags = pacf_lags, train_data = None, 
                      test_data = None, train_rs = None, test_rs = None, acf_max_df = None, 
                      pacf_max_df = None,garch_data = None, file_path = file_path,
                      arima_file_path = arima_file_path , res_data = None)
                    
    obj.read_data()
    obj.return_series()
    obj.lag_cal()
    print(obj.gr_model())
    # obj.gr_prediction()
