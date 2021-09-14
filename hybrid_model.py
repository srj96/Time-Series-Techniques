from arch import arch_model 
import pandas as pd 
import numpy  as np
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf

class HybridModel:

    '''Below calculated values : Return Series , PACF and ACF , GARCH Prediction , Transformation '''

    def __init__(self,**kwargs):
        attr_list = set(['train_path','train_data','vald_data','train_rs','vald_rs',
                         'vald_path','rlags','acf_lags','pacf_lags','acf_max_df',
                         'pacf_max_df','garch_data','file_path','arima_data_new',
                         'arima_file_path', 'arima_path_new'])
        [self.__setattr__(k,kwargs[k]) for k in kwargs.keys() if k in attr_list ]
    
    def read_data(self):
        with open(self.train_path, 'r') as train_file:
            self.train_data = pd.read_csv(train_file)
        with open(self.vald_path, 'r') as vald_file:
            self.vald_data = pd.read_csv(vald_file)
        
        self.train_data = self.train_data.reset_index()
        self.train_data = self.train_data[['date', 'period', 'price']]

        self.vald_data = self.vald_data.reset_index()
        self.vald_data = self.vald_data[['date', 'period' , 'price']]

        return(self.train_data,self.vald_data)

    # Return Series with shift given in the main() < presently calculated for t-1>
    def return_series(self):
        self.train_rs = pd.DataFrame()
        for ix in self.train_data['period'].unique():
            rs_val = np.log(self.train_data[self.train_data['period'] == ix]['price'] / self.train_data[self.train_data['period'] == ix]['price'].shift(periods = self.rlags, fill_value = 0))
            p = {'date' : self.train_data[self.train_data['period'] == ix]['date'], 'period' : ix , 'lag_{}'.format(self.rlags) : rs_val}
            df = pd.DataFrame(data = p)
            self.train_rs = self.train_rs.append(df)
        
        self.vald_rs = pd.DataFrame()
        for iy in self.vald_data['period'].unique():
            rs_val_new = np.log(self.vald_data[self.vald_data['period'] == iy]['price'] / self.vald_data[self.vald_data['period'] == iy]['price'].shift(periods = self.rlags, fill_value = 0))
            p = {'date' : self.vald_data[self.vald_data['period'] == iy]['date'], 'period' : iy , 'lag_{}'.format(self.rlags) : rs_val_new}
            df_new = pd.DataFrame(data = p)
            self.vald_rs = self.vald_rs.append(df_new)

        return(self.train_rs,self.vald_rs) 
    
    # PACF and ACF with number of lags given in main() < presently given lags = 30>
    def lag_cal(self):

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
    
    # Arima model implementation for new validation set data which will be used further for 
    # Garch model
    def ar_model_new(self):
        train_rs_ar = self.train_rs[self.train_rs['lag_1'] != np.inf]
        self.arima_data_new = pd.DataFrame()
        for ix in sorted(self.train_rs['period'].unique()):
            train = train_rs_ar[train_rs_ar['period'] == ix]['lag_1']
            vald = self.vald_rs[self.vald_rs['period'] == ix]['lag_1']
            lag_q = [q for iq , q in enumerate(self.acf_max_df[self.acf_max_df['period'] == ix]['lag_acf'])][0]
            lag_p = [p for ip , p in enumerate(self.pacf_max_df[self.pacf_max_df['period']== ix]['lag_pcf'])][0]
    
            model = ARIMA(train, order = (lag_p,0,lag_q))
    
            arm = model.fit(disp = 0)
            output = arm.forecast(steps = len(vald))
            yhat = output[0]
            print('time_slot',ix)
            p = {'date': self.vald_rs[self.vald_rs['period'] == ix]['date'],'period': ix , 'pred_val': yhat}
            df_1 = pd.DataFrame(data = p)
            self.arima_data_new= self.arima_data_new.append(df_1)
        return(self.arima_data_new)
    
    # Log transformation to exponentiation to generate the results for residual error
    def ar_pred_new(self):
        self.arima_data_new = self.arima_data_new.reset_index()
        self.arima_data_new = self.arima_data_new[['date','period','pred_val']]

        arima_data_mod = pd.merge(self.arima_data_new,self.vald_data , on=['date','period'])
        pred_price = np.exp(arima_data_mod['pred_val'])*(arima_data_mod['price'].shift(periods = 1 , fill_value = 0))
        arima_data_mod['predicted_price'] = pred_price
        arima_data_pred = arima_data_mod[arima_data_mod['date'] != '2019-05-31']
        error = abs(arima_data_pred['price'] - arima_data_pred['predicted_price'])
        ape = ((error)/arima_data_pred['price'])*100 
        arima_data_pred['Res_Error'] = ape
        arima_data_pred = arima_data_pred.rename(columns = {'price' : 'actual_price'})
        arima_data_pred = arima_data_pred.round({'Res_Error' : 2})
        arima_data_pred = arima_data_pred.round({'predicted_price' : 2})
        
        with open(self.arima_path_new,'w') as savefile:
            arima_data_pred.to_csv(savefile)
        
        print(arima_data_pred)
        return(arima_data_pred)


    
    # Garch Model predictions with values of p and q picked from lag_cal() and assumes the input
    # series as residual errors from the Arima result
    def gr_model(self):
        with open(self.arima_path_new,'r') as readfile:
            res_data = pd.read_csv(readfile)
        res_df = res_data[['date','period','predicted_price']]
        # res_df['residual_variance'] = res_df['predicted_price'].pct_change(periods = 1)
        # res_df['residual_variance'] = res_df['predicted_price']/res_df['predicted_price'].shift(periods = 1, fill_value = 0)
        # res_df = res_df[res_df['residual_variance'] != np.inf]
        # res_df = res_df.dropna() 
        #------------------------------------------------------------------------------
        self.vald_rs['date'] = pd.to_datetime(self.vald_rs['date'])
        
        self.garch_data = pd.DataFrame()
        for ix in sorted(self.train_rs['period'].unique()):
            val = self.vald_rs[self.vald_rs['period'] == ix]['lag_1']
            lag_q = [q for iq , q in enumerate(self.acf_max_df[self.acf_max_df['period'] == ix]['lag_acf'])][0]
            lag_p = [p for ip , p in enumerate(self.pacf_max_df[self.pacf_max_df['period']== ix]['lag_pcf'])][0]
            garch_train = 100*(res_df[res_df['period'] == ix]['predicted_price'].pct_change(periods = 1).dropna())

            model = arch_model(garch_train, mean = 'Zero', vol = 'GARCH', p = lag_p, q = lag_q)
            garch = model.fit()
            output = garch.forecast(horizon = len(val))
            print('time_slot',ix)
            var = output.variance.values[-1,:]
            std_dev = 0.01*np.sqrt(output.variance.values[-1,:]) 
            print(var)
            print(std_dev) 
            
            p = {'date': self.vald_rs[self.vald_rs['period'] == ix]['date'],'period': ix , 'variance': var , 'std_deviation' : std_dev}
            df_1 = pd.DataFrame(data = p)
            self.garch_data = self.garch_data.append(df_1)
        print(self.garch_data)
        return(self.garch_data) 
            
    # -------------------------------------------------------------------------------------------
    
    # Prediction is based on standard deviation calculated from the Garch Model and using it to 
    # predict the future values for (Aug-Sept). Basically predicted price is taken as mean (mu)
    # and thus for confidence interval 95% we have mu + 2*sigma and mu - 2*sigma, where sigma
    # is standard deviation. The standard deviations are divided into two scenarios
    # from a) Residual Variances and b) Return Series 
    
    
    def hybrid_prediction(self):
        with open(arima_file_path,'r') as read_file:
            orig_test_data = pd.read_csv(read_file)
        orig_test_df = orig_test_data[['date','period','actual_price','predicted_price']]
        orig_test_df['date'] = pd.to_datetime(orig_test_df['date'])
        # self.garch_data = self.garch_data.reset_index() 
        #------------------------------------------------------------------------------
        # Prediction from the residual variances #
        self.garch_data = self.garch_data[['date','period','variance','std_deviation']]
        date_list = self.garch_data['date'].unique()
        self.garch_data = self.garch_data[self.garch_data['date'] != date_list[0]]
        self.garch_data['date'] = pd.to_datetime(self.garch_data['date'])
        print(len(self.garch_data['date'].unique()))
        if len(self.garch_data['date'].unique())%2 == 0 :
            add_offset = len(self.garch_data['date'].unique())/30 
        else :
            add_offset = (len(self.garch_data['date'].unique())-1)/30
        self.garch_data['date'] = self.garch_data['date'] + pd.offsets.MonthOffset(add_offset) 
        self.garch_data = self.garch_data[['date','period','variance','std_deviation']]
        garch_data_full = pd.merge(orig_test_df, self.garch_data, on = ['date','period'])  

        garch_data_full = garch_data_full.round({'predicted_price' : 2 ,'variance' : 2 , 'std_deviation' : 2})

        
        garch_data_full['lower_bound_predicted_price'] = garch_data_full['predicted_price'] - (2*(garch_data_full['std_deviation']))
        garch_data_full['upper_bound_predicted_price'] = garch_data_full['predicted_price'] + (2*(garch_data_full['std_deviation']))

        with open(file_path, 'w') as save_pred_file:
            garch_data_full.to_csv(save_pred_file)
        
        return(garch_data_full)
    
      

if __name__ == '__main__':
    
    # Train Validation files 
    train_path = '/home/congnitensor/Python_projects/ptc_armax/train_val_test/train_ptc_e1.csv'
    vald_path = '/home/congnitensor/Python_projects/ptc_armax/train_val_test/vald_ptc_e1.csv'
    
    # Save the new prediction file 
    # file_path = '/home/congnitensor/Python_projects/ptc_armax/ptc_hybrid_pred/ptc_hybrid_pred_e1.csv'
    file_path = '/home/congnitensor/Python_projects/ptc_armax/ptc_hybrid_pred/ptc_hybrid_pred_new_e1.csv'

    # Saves the new validation set data from arima and will be taken for further prediction
    arima_path_new = '/home/congnitensor/Python_projects/ptc_armax/ptc_arima_new_file/ptc_arima_new_e1.csv'
    
    # Picks the original test file from previous arima model run (Aug-Sept)
    arima_file_path = '/home/congnitensor/Python_projects/ptc_armax/ptc_area_pred/ptc_area_e1_pred.csv'
    rlags = 1
    acf_lags = 30
    pacf_lags = 30
    obj = HybridModel(train_path = train_path, vald_path = vald_path , rlags = rlags, 
                      acf_lags = acf_lags, pacf_lags = pacf_lags,train_data = None, 
                      vald_data = None, train_rs = None, vald_rs = None, 
                      acf_max_df = None, pacf_max_df = None,garch_data = None, 
                      file_path = file_path,arima_file_path = arima_file_path , 
                      arima_path_new = arima_path_new , arima_data_new = None)
                    
    obj.read_data()
    obj.return_series()
    obj.lag_cal()
    # obj.ar_model_new()
    # obj.ar_pred_new()
    
    obj.gr_model()
    print(obj.hybrid_prediction())
    