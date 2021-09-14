import pandas as pd 
import numpy as np 
from sklearn.utils import resample
from sklearn_extensions.extreme_learning_machines.elm import ELMRegressor
from sklearn.preprocessing import StandardScaler
from arch import arch_model
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf

class ELMarch:

    def __init__(self,**kwargs):

        attr_list = set(['train_path','train_label','train_feature','val_path','val_label',
                         'val_feature','train_label_rs','val_label_rs','rlags','acf_lags','pacf_lags',
                         'acf_max_df','pacf_max_df','elm_data','elm_data_pred','garch_data',
                         'col_names','file_path', 'test_path'])
        
        [self.__setattr__(k,kwargs[k]) for k in kwargs.keys() if k in attr_list]

    # Distribution of features and labels
    def feature_eng(self):
        with open(self.train_path, 'r') as train_file:
            train_data = pd.read_csv(train_file)
        with open(self.val_path, 'r') as val_file:
            val_data = pd.read_csv(val_file)

        feature_col = [col for col in train_data.columns if col not in self.col_names]
        # feature_col = [col for col in train_data.columns if col in self.col_names]

        self.train_feature = train_data[feature_col]
        self.train_label = train_data[['date','period','price']]
        self.val_feature = val_data[feature_col]
        self.val_label = val_data[['date','period','price']]
        
        print(list(self.train_feature))

        return(self.train_feature,self.train_label,self.val_feature,self.val_label)

    
    # Return Series for Train and Test values 
    def return_series(self):
        self.train_label_rs = pd.DataFrame()
        for ix in sorted(self.train_label['period'].unique()):
            rs_val_1 = np.log(self.train_label[self.train_label['period'] == ix]['price'] / self.train_label[self.train_label['period'] == ix]['price'].shift(periods = self.rlags, fill_value = 0))
            p = {'date' : self.train_label[self.train_label['period'] == ix]['date'], 'period' : ix , 'lag_{}'.format(self.rlags) : rs_val_1}
            df_1 = pd.DataFrame(data = p)
            self.train_label_rs = self.train_label_rs.append(df_1)
        
        self.val_label_rs = pd.DataFrame()
        for iy in sorted(self.val_label['period'].unique()):
            rs_val_2 = np.log(self.val_label[self.val_label['period'] == iy]['price'] / self.val_label[self.val_label['period'] == iy]['price'].shift(periods = self.rlags, fill_value = 0))
            p = {'date' : self.val_label[self.val_label['period'] == iy]['date'], 'period' : iy , 'lag_{}'.format(self.rlags) : rs_val_2}
            df_2 = pd.DataFrame(data = p)
            self.val_label_rs = self.val_label_rs.append(df_2)

        return(self.train_label_rs,self.val_label_rs)
    
    # PACF and ACF calculation for p and q respectively 
    def lag_cal(self):

        train_rs_cf = self.train_label_rs[self.train_label_rs['lag_1'] != np.inf]
        self.acf_max_df = pd.DataFrame()
        for ix in self.train_label_rs['period'].unique():
            acf_val = acf(train_rs_cf[train_rs_cf['period'] == ix]['lag_1'], nlags= self.acf_lags, fft = False)
            acf_max_val = max(abs(acf_val[acf_val != 1.0]))
            acf_list = list(abs(acf_val))
            acf_max_list = acf_list.index(acf_max_val)
            p = {'period': ix , 'lag_acf' : acf_max_list}
            df_1 = pd.DataFrame(data = p, index = [0])
            self.acf_max_df = self.acf_max_df.append(df_1)
    
    
        self.pacf_max_df = pd.DataFrame()
        for iy in self.train_label_rs['period'].unique():
            pacf_val = pacf(train_rs_cf[train_rs_cf['period'] == iy]['lag_1'], nlags= self.pacf_lags)
            pacf_max_val = max(abs(pacf_val[pacf_val != 1.0]))
            pacf_list = list(abs(pacf_val))
            pacf_max_list = pacf_list.index(pacf_max_val)
            p = {'period' : iy , 'lag_pcf' : pacf_max_list}
            df_2 = pd.DataFrame(data = p, index = [0]) 
            self.pacf_max_df = self.pacf_max_df.append(df_2)
        
        return(list(self.acf_max_df['lag_acf']),list(self.pacf_max_df['lag_pcf']))
    
    def elm_model(self):
        train_label_elm = self.train_label_rs[self.train_label_rs['lag_1'] != np.inf]
        self.train_feature = self.train_feature.drop(columns = ['area','Unnamed: 0', 'level_0', 'Unnamed: 0.1', 'index'])
        self.val_feature = self.val_feature.drop(columns = ['area','Unnamed: 0', 'level_0', 'Unnamed: 0.1', 'index'])
        self.train_feature = self.train_feature[self.train_feature['date'] != '2016-01-01']
        print(self.train_feature.shape)
        print(train_label_elm.shape) 
        stdsc = StandardScaler()
        elmr = ELMRegressor()
        self.elm_data = pd.DataFrame()
        for ix in self.train_label_rs['period'].unique():
            train_label_data = train_label_elm[train_label_elm['period'] == ix]['lag_1']
            train_feature_data = self.train_feature[self.train_feature['period'] == ix][[col for col in self.train_feature.columns if col not in ['date','period']]]
            val_feature_data = self.val_feature[self.val_feature['period'] == ix][[col for col in self.val_feature.columns if col not in ['date','period']]] 
            train_feature_data = stdsc.fit_transform(train_feature_data)
            print(train_feature_data.shape) 
            print(train_label_data.shape)
            val_feature_data = stdsc.fit_transform(val_feature_data)
            model = elmr.fit(train_feature_data,train_label_data)
            yhat = model.predict(val_feature_data)
            print('time_slot:', ix) 
            p = {'date': self.val_label_rs[self.val_label_rs['period'] == ix]['date'],'period': ix , 'pred_val': yhat}
            df_1 = pd.DataFrame(data = p)
            self.elm_data = self.elm_data.append(df_1)
        return(self.elm_data)
    
    def elm_prediction(self):
        self.elm_data = self.elm_data.reset_index()
        self.elm_data = self.elm_data[['date','period','pred_val']]

        elm_data_full = pd.merge(self.elm_data,self.val_label , on=['date','period'])
        pred_price = np.exp(elm_data_full['pred_val'])*(elm_data_full['price'].shift(periods = 1 , fill_value = 0))
        elm_data_full['predicted_price'] = pred_price
        self.elm_data_pred = elm_data_full[elm_data_full['date'] != '2019-05-31']
        error = abs(self.elm_data_pred['price'] - self.elm_data_pred['predicted_price'])
        ape = ((error)/self.elm_data_pred['price'])*100 
        self.elm_data_pred['%_Error'] = ape
        self.elm_data_pred = self.elm_data_pred.rename(columns = {'price' : 'actual_price'})
        self.elm_data_pred = self.elm_data_pred.round({'%_Error' : 2 , 'predicted_price' : 2})
        
        print(self.elm_data_pred)

        return(self.elm_data_pred)
    
    # Garch Model Implementation using residual variances 
    def gr_model(self):
        elm_df = self.elm_data_pred[['period','predicted_price']]
        elm_df['residual_variance'] = 100*elm_df['predicted_price'].pct_change(periods = 1)
        elm_df = elm_df.dropna()
        print(elm_df)

        self.val_label_rs['date'] = pd.to_datetime(self.val_label_rs['date'])
        
        self.garch_data = pd.DataFrame()
        for ix in sorted(self.train_label_rs['period'].unique()):
            val = self.val_label_rs[self.val_label_rs['period'] == ix]['lag_1']
            lag_q = [q for iq , q in enumerate(self.acf_max_df[self.acf_max_df['period'] == ix]['lag_acf'])][0]
            lag_p = [p for ip , p in enumerate(self.pacf_max_df[self.pacf_max_df['period']== ix]['lag_pcf'])][0]
            garch_train = elm_df[elm_df['period'] == ix]['residual_variance']
            model = arch_model(garch_train, mean = 'Zero', vol = 'GARCH', p = lag_p, q = lag_q)
            garch = model.fit()
            output = garch.forecast(horizon = len(val))
            print('time_slot',ix)
            var = output.variance.values[-1,:]
            std_dev = np.sqrt(output.variance.values[-1,:])
            # print(var)
            # print(std_dev) 
            p = {'date': self.val_label_rs[self.val_label_rs['period'] == ix]['date'],'period': ix , 'variance': var , 'std_deviation' : std_dev}
            df_1 = pd.DataFrame(data = p)
            self.garch_data = self.garch_data.append(df_1)
        print(self.garch_data)
        return(self.garch_data) 
    
    def elm_garch_pred(self):
        with open(self.test_path,'r') as test_file:
            test_data = pd.read_csv(test_file)
        test_df = test_data[['date','period','actual_price','predicted_price']]
        test_df['date'] = pd.to_datetime(test_df['date'])
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
        elm_garch_data_full = pd.merge(test_df, self.garch_data, on = ['date','period'])  

        elm_garch_data_full = elm_garch_data_full.round({'predicted_price' : 2 ,'variance' : 2 , 'std_deviation' : 2})

        
        elm_garch_data_full['lower_bound_predicted_price'] = elm_garch_data_full['predicted_price'] - (2*(elm_garch_data_full['std_deviation']))
        elm_garch_data_full['upper_bound_predicted_price'] = elm_garch_data_full['predicted_price'] + (2*(elm_garch_data_full['std_deviation']))

        with open(file_path, 'w') as save_pred_file:
            elm_garch_data_full.to_csv(save_pred_file)
        
        print(elm_garch_data_full)
        return(elm_garch_data_full)



if __name__ == "__main__":

    train_path = '/home/congnitensor/Python_projects/ptc_armax/train_test_elm/train_ptc_e1_new.csv'
    val_path =  '/home/congnitensor/Python_projects/ptc_armax/train_test_elm/val_ptc_e1.csv'
    test_path =  '/home/congnitensor/Python_projects/ptc_armax/ptc_elm_pred/ptc_elm_pred_e1_full.csv'
    file_path =  '/home/congnitensor/Python_projects/ptc_armax/ptc_elm_pred/ptc_elm_pred_e1_new_full.csv'

    r_lags = 1
    acf_lags = 30
    pacf_lags = 30

    col_names = ['price']

    obj = ELMarch(train_path = train_path,train_label = None,
                  train_feature = None,val_path = val_path,
                  val_label = None,val_feature = None,
                  train_label_rs = None,val_label_rs = None,
                  rlags = r_lags,acf_lags = acf_lags,pacf_lags = pacf_lags,
                  acf_max_df = None,pacf_max_df = None,elm_data = None,
                  elm_data_pred = None,garch_data = None,
                  col_names = col_names,file_path = file_path, test_path = test_path)
    
    obj.feature_eng()
    obj.return_series()
    obj.lag_cal()
    obj.elm_model()
    obj.elm_prediction()
    obj.gr_model()
    obj.elm_garch_pred()



