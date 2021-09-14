import pandas as pd 
import numpy as np 
from sklearn.utils import resample 
from sklearn_extensions.extreme_learning_machines.elm import ELMRegressor
from sklearn.preprocessing import StandardScaler

class ExtremeLM:
    
    def __init__(self,**kwargs):
        
        [self.__setattr__(k,kwargs[k]) for k in kwargs.keys()] 
    
    def feature_eng(self):
        with open(self.train_path, 'r') as train_file:
            train_data = pd.read_csv(train_file)
        with open(self.test_path, 'r') as test_file:
            test_data = pd.read_csv(test_file)

        # feature_col = [col for col in train_data.columns if col not in self.col_names]
        feature_col = [col for col in train_data.columns if col in self.col_names]

        self.train_feature = train_data[feature_col]
        self.train_label = train_data[['date','period','price']]
        self.test_feature = test_data[feature_col]
        self.test_label = test_data[['date','period','price']]
        
        print(list(self.train_feature))
        print(list(self.test_feature))

        return(self.train_feature,self.train_label,self.test_feature,self.test_label)
    
    def elm_model(self):
        # self.train_feature = self.train_feature.drop(columns = ['area','Unnamed: 0', 'Unnamed: 0.1', 'index'])
        # self.test_feature = self.test_feature.drop(columns = ['area','Unnamed: 0', 'Unnamed: 0.1', 'index'])

        print(self.train_feature.shape)
        stdsc = StandardScaler()
        elmr = ELMRegressor()
        self.elm_data = pd.DataFrame()
        for ix in sorted(self.train_label['period'].unique()):
            train_label_data = self.train_label[self.train_label['period'] == ix]['price']
            train_feature_data = self.train_feature[self.train_feature['period'] == ix][[col for col in self.train_feature.columns if col not in ['date','period']]]
            test_feature_data = self.test_feature[self.test_feature['period'] == ix][[col for col in self.test_feature.columns if col not in ['date','period']]] 
            
            train_feature_data = stdsc.fit_transform(train_feature_data)
            test_feature_data = stdsc.fit_transform(test_feature_data)

            print(train_feature_data.shape) 
            print(train_label_data.shape)

            model = elmr.fit(train_feature_data,train_label_data)
            yhat = model.predict(test_feature_data)
            print('time_slot:', ix) 
            p = {'date': self.test_label[self.test_label['period'] == ix]['date'], 'period' : ix ,'actual_price' : self.test_label[self.test_label['period'] == ix]['price'], 'predicted_price': yhat}
            df_1 = pd.DataFrame(data = p)
            self.elm_data = self.elm_data.append(df_1)
        return(self.elm_data)

    def elm_prediction(self):
        self.elm_data = self.elm_data.reset_index()
        self.elm_data = self.elm_data[['date','period','actual_price','predicted_price']]

        error = abs(self.elm_data['actual_price'] - self.elm_data['predicted_price'])
        res_error = ((error)/self.elm_data['actual_price'])
        ape = res_error*100
        self.elm_data['%_Error'] = ape
        self.elm_data['Residual_Error'] = res_error

        self.elm_data = self.elm_data.round({'%_Error' : 2 , 'predicted_price' : 2})
        
        print(self.elm_data)
        with open(self.file_path,'w') as savefile:
            self.elm_data.to_csv(savefile)
        
        return(self.elm_data)  




if __name__ == "__main__":

    train_path = '/home/congnitensor/Python_projects/ptc_armax/train_test_elm/train_ptc_e1.csv'
    test_path =  '/home/congnitensor/Python_projects/ptc_armax/train_test_elm/test_ptc_e1.csv'
    file_path =  '/home/congnitensor/Python_projects/ptc_armax/ptc_elm_boostrap_pred/ptc_elm_pred_e1_boot_strap_volume.csv'


    # To be taken column features // unamed index not to be taken //
    col_names = ['date','period','buy_quantity','sell_quantity'] 

    # Not to be taken column features --> Weather Features only // drop unamed
    # col_names = ['price','buy_quantity', 'sell_quantity']

    # Not to be taken column feature --> Quantity + Weather both // drop unamed 
    # col_names = ['price'] 

    obj = ExtremeLM(train_path = train_path, test_path = test_path, file_path = file_path,
                   train_label = None, train_feature = None, test_label = None, test_feature = None,
                   elm_data = None, col_names = col_names)
    
    print(obj.feature_eng())
    obj.elm_model()
    obj.elm_prediction()

    

