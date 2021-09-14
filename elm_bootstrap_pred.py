import pandas as pd 
import numpy as np 
from sklearn.utils import resample 
from sklearn_extensions.extreme_learning_machines.elm import ELMRegressor
from sklearn.preprocessing import StandardScaler

class ExtremeLMBootstrap:
    def __init__(self,**kwargs):

        [self.__setattr__(k,kwargs[k]) for k in kwargs.keys()]
    
    def data_preproc(self):
        with open(self.elm_path, 'r') as read_file:
            self.elm_df = pd.read_csv(read_file)
        
        #recenter the errors 
        self.res_error = self.elm_df['Residual_Error'] - self.elm_df['Residual_Error'].mean()
        
        print(len(self.res_error))
        return(self.res_error)

    
    def elm_bootstrap_model(self):
        
        with open(self.train_path, 'r') as train_path:
            train_data = pd.read_csv(train_path)
        with open(self.test_path, 'r') as test_path:
            test_data = pd.read_csv(test_path)
        
        stdsc = StandardScaler()
        elmr = ELMRegressor() 
        
        # feature_col = [col for col in test_data.columns if col not in self.col_names]
        feature_col = [col for col in train_data.columns if col in self.col_names]
        
        test_features =  test_data.loc[:,feature_col]

        period_list = sorted(train_data['period'].unique())
        print(period_list) 

        n_iteration = self.iteration 

        
        # test_features = test_features.drop(columns = ['area','time','Unnamed: 0','Unnamed: 0.1', 'index'])
        col  = [c for c in test_features.columns if c not in ['date','period']]
        
        self.stats = []
        for n_iter in range(n_iteration) :
            n_pred = []
            for ix in period_list:
                boot_error = resample(self.res_error, replace = True, n_samples = len(self.res_error), random_state = None) 
                boot_error = boot_error.tolist()
                self.elm_df['new_label'] = self.elm_df['predicted_price'] + boot_error
                label = self.elm_df.loc[self.elm_df['period'] == ix,'new_label']
                test_features_data = test_features.loc[test_features['period'] == ix , col] 
                test_features_data = stdsc.fit_transform(test_features_data)
                model = elmr.fit(test_features_data,label)
                yhat = model.predict(test_features_data)
                print('time_slot:', ix)
                print(test_features_data.shape)
                print(label.shape)
                n_pred.append(yhat) 

            self.stats.append(n_pred) 
        print('the set of new data',self.stats) 
    
    def elm_bootstrap_pred(self):
        alpha = 0.05 
        p_lower = (alpha/2)
        p_upper = (1-(alpha/2))
        print(p_lower)
        print(p_upper)
        lower_pos = int(self.iteration*p_lower)
        upper_pos = int(self.iteration*p_upper)

        lower_pred = self.stats[lower_pos]
        upper_pred = self.stats[upper_pos]
 
        print(lower_pos)
        print(upper_pos)

        print(len(lower_pred))
        print(len(upper_pred))

        # lp = [l for il,l in enumerate(lower_pred)][]
        # up = [u for iu,u in enumerate(upper_pred)][]
        # print('lower',lp)
        # print('upper',up)
        # print('the lower predicted price : ',lower_df)
        # print('the upper predicted price :',upper_df)

        print('the lower predicted list : ',lower_pred)
        print('the upper predicted list :',upper_pred)

        elm_bstrap = pd.DataFrame()
        for ix in range(0,len(lower_pred)):
            p = {'lower_predicted_price': lower_pred[ix], 'upper_predicted_price': upper_pred[ix]}
            
            df = pd.DataFrame(data = p)
            elm_bstrap = elm_bstrap.append(df)

        print(elm_bstrap)

        elm_bstrap = elm_bstrap.round({'lower_predicted_price' : 2 , 'upper_predicted_price' : 2})

        with open(self.file_path,'w') as file_path:
            elm_bstrap.to_csv(file_path) 
            

if __name__ == "__main__":

    train_path = '/home/congnitensor/Python_projects/ptc_armax/train_test_elm/train_ptc_e1.csv'
    test_path =  '/home/congnitensor/Python_projects/ptc_armax/train_test_elm/test_ptc_e1.csv'
    elm_path =  '/home/congnitensor/Python_projects/ptc_armax/ptc_elm_boostrap_pred/ptc_elm_pred_e1_boot_strap_volume.csv'
    file_path = '/home/congnitensor/Python_projects/ptc_armax/ptc_elm_boostrap_pred/ptc_elm_pred_e1_boot_strap_prediction_price_volume.csv'

    n_iteration = 100

    # To be taken column features // unamed index not to be taken //
    col_names = ['date','period','buy_quantity','sell_quantity'] 

    # Not to be taken column features --> Weather Features only // drop unamed
    # col_names = ['price','buy_quantity', 'sell_quantity']

    # Not to be taken column feature --> Quantity + Weather both // drop unamed 
    # col_names = ['price']

    obj = ExtremeLMBootstrap(train_path = train_path, test_path = test_path, elm_df = None,
                             res_error = None, col_names = col_names, iteration = n_iteration,
                             elm_path = elm_path, elm_data = None, file_path = file_path, stats = None)
        
    
    obj.data_preproc()
    obj.elm_bootstrap_model()
    obj.elm_bootstrap_pred()