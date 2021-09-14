import pandas as pd 
import numpy as np 
from sklearn_extensions.extreme_learning_machines.elm import ELMRegressor
from sklearn.preprocessing import StandardScaler

class ExtremeLMod:
    
    def __init__(self,**kwargs):

        attr_list = set(['val_path','val_label','val_feature','test_path','test_label',
                         'test_feature','val_label_rs','test_label_rs','rlags','elm_data',
                         'col_names','file_path'])
        
        [self.__setattr__(k,kwargs[k]) for k in kwargs.keys() if k in attr_list]
    
    def feature_eng(self):
        with open(self.val_path, 'r') as val_file:
            val_data = pd.read_csv(val_file)
        with open(self.test_path, 'r') as test_file:
            test_data = pd.read_csv(test_file)

        feature_col = [col for col in val_data.columns if col not in self.col_names]
        # feature_col = [col for col in train_data.columns if col in self.col_names]

        self.val_feature = val_data[feature_col]
        self.val_label = val_data[['date','period','price']]
        self.test_feature = test_data[feature_col]
        self.test_label = test_data[['date','period','price']]
        
        print(list(self.train_feature))

        return(self.val_feature,self.val_label,self.test_feature,self.test_label)
    
    def return_series(self):
        self.val_label_rs = pd.DataFrame()
        for ix in sorted(self.val_label['period'].unique()):
            rs_val_1 = np.log(self.val_label[self.val_label['period'] == ix]['price'] / self.val_label[self.val_label['period'] == ix]['price'].shift(periods = self.rlags, fill_value = 0))
            p = {'date' : self.val_label[self.val_label['period'] == ix]['date'], 'period' : ix , 'lag_{}'.format(self.rlags) : rs_val_1}
            df_1 = pd.DataFrame(data = p)
            self.val_label_rs = self.val_label_rs.append(df_1)
        
        self.test_label_rs = pd.DataFrame()
        for iy in sorted(self.test_label['period'].unique()):
            rs_val_2 = np.log(self.test_label[self.test_label['period'] == iy]['price'] / self.test_label[self.test_label['period'] == iy]['price'].shift(periods = self.rlags, fill_value = 0))
            p = {'date' : self.test_label[self.test_label['period'] == iy]['date'], 'period' : iy , 'lag_{}'.format(self.rlags) : rs_val_2}
            df_2 = pd.DataFrame(data = p)
            self.test_label_rs = self.test_label_rs.append(df_2)

        return(self.train_label_rs,self.test_label_rs)
    
    def elm_model(self):
        train_label_elm = self.train_label_rs[self.train_label_rs['lag_1'] != np.inf]
        self.train_feature = self.train_feature.drop(columns = ['area','Unnamed: 0', 'level_0', 'Unnamed: 0.1', 'index'])
        self.train_feature = self.train_feature[self.train_feature['date'] != '2016-01-01']
        self.test_feature = self.test_feature.drop(columns = ['area','Unnamed: 0', 'level_0', 'Unnamed: 0.1', 'index'])

        print(self.train_feature.shape)
        print(train_label_elm.shape) 
        stdsc = StandardScaler()
        elmr = ELMRegressor()
        self.elm_data = pd.DataFrame()
        for ix in self.train_label_rs['period'].unique():
            train_label_data = train_label_elm[train_label_elm['period'] == ix]['lag_1']
            train_feature_data = self.train_feature[self.train_feature['period'] == ix][[col for col in self.train_feature.columns if col not in ['date','period']]]
            test_feature_data = self.test_feature[self.test_feature['period'] == ix][[col for col in self.test_feature.columns if col not in ['date','period']]] 
            train_feature_data = stdsc.fit_transform(train_feature_data)
            print(train_feature_data.shape) 
            print(train_label_data.shape)
            test_feature_data = stdsc.fit_transform(test_feature_data)
            model = elmr.fit(train_feature_data,train_label_data)
            yhat = model.predict(test_feature_data)
            print('time_slot:', ix) 
            p = {'date': self.test_label_rs[self.test_label_rs['period'] == ix]['date'],'period': ix , 'pred_val': yhat}
            df_1 = pd.DataFrame(data = p)
            self.elm_data = self.elm_data.append(df_1)
        return(self.elm_data)

    def elm_prediction(self):
        self.elm_data = self.elm_data.reset_index()
        self.elm_data = self.elm_data[['date','period','pred_val']]

        elm_data_full = pd.merge(self.elm_data,self.test_label , on=['date','period'])
        pred_price = np.exp(elm_data_full['pred_val'])*(elm_data_full['price'].shift(periods = 1 , fill_value = 0))
        elm_data_full['predicted_price'] = pred_price
        elm_data_pred = elm_data_full[elm_data_full['date'] != '2019-07-31']
        error = abs(elm_data_pred['price'] - elm_data_pred['predicted_price'])
        ape = ((error)/elm_data_pred['price'])*100 
        elm_data_pred['%_Error'] = ape
        elm_data_pred = elm_data_pred.rename(columns = {'price' : 'actual_price'})
        elm_data_pred = elm_data_pred.round({'%_Error' : 2 , 'predicted_price' : 2})
        
        print(elm_data_pred)
        with open(self.file_path,'w') as savefile:
            elm_data_pred.to_csv(savefile)
        
        return(elm_data_pred)




if __name__ == "__main__":

    train_path = '/home/congnitensor/Python_projects/ptc_armax/train_test_elm/train_ptc_e1.csv'
    test_path =  '/home/congnitensor/Python_projects/ptc_armax/train_test_elm/test_ptc_e1.csv'
    file_path =  '/home/congnitensor/Python_projects/ptc_armax/ptc_elm_pred/ptc_elm_pred_e1_full.csv'

    r_lags = 1

    # To be taken column features // unamed index not to be taken //
    # col_names = ['date','period','buy_quantity','sell_quantity'] 

    # Not to be taken column features --> Weather Features only // drop unamed
    # col_names = ['price','buy_quantity', 'sell_quantity']

    # Not to be taken column feature --> Quantity + Weather both // drop unamed 
    col_names = ['price']

    obj = ExtremeLMod(train_path = train_path, test_path = test_path, file_path = file_path,
                    train_data = None, test_data = None, train_label_rs = None, test_label_rs = None,
                    rlags = r_lags,elm_data = None, col_names = col_names)
    
    print(obj.feature_eng())
    obj.return_series()
    obj.elm_model()
    obj.elm_prediction()

    
