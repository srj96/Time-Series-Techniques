import pandas as pd 
import numpy as np 

class Preprocess:
    '''Below the data is split into train and test and then groups it according to
    time periods. The start date and end date are given in the main()'''
    
    def __init__(self,**kwargs):
        attr_list = set(['path','seg_date','train_date','vald_date','train_path','vald_path'])
        [self.__setattr__(k , kwargs[k]) for k in kwargs.keys() if k in attr_list]
        
        
    def data_split(self):
        df = pd.read_csv(self.path)
        df = df.rename(columns = {'mcp_price' : 'price'})
        df_new = df[['date','price','period']]
        df_1619 = df_new.loc[(df_new['date'] >= self.seg_date[0]) & (df_new['date'] <= self.seg_date[-1])]
        df_1619= df_1619.sort_values(by = ['date'])
        df_1619 = df_1619.reset_index()
        df_1619 = df_1619[['date','period','price']]

        # df_1619 = df_1619[df_1619['price'] != '']
        # df_1619['price'] = df_1619['price'].dropna()
        # df_1619 = df_1619.dropna(axis = 0)

        df_east= pd.DataFrame()
        for ix in df_1619['period'].unique():
            y = df_1619.loc[df_1619['period'] == ix][['date', 'period', 'price']]
            df_east = df_east.append(y)   
            train_set = df_east.loc[(df_east['date'] >= self.train_date[0]) & (df_east['date'] <= self.train_date[-1])]
            test_set = df_east.loc[(df_east['date'] >= self.test_date[0]) & (df_east['date'] <= self.vald_date[-1])]
            with open(self.train_path, 'w') as train_file:
                train_set.to_csv(train_file)
            with open(self.vald_path, 'w') as vald_file:
                test_set.to_csv(vald_file) 

        return(test_set,train_set) 

if __name__ == '__main__':
    
    file_path = '/home/congnitensor/Python_projects/ptc_armax/ptc_area_data/ptc_base.csv'
    train_path = '/home/congnitensor/Python_projects/ptc_armax/train_val_test/train_ptc_base.csv'
    vald_path = '/home/congnitensor/Python_projects/ptc_armax/train_val_test/vald_ptc_base.csv'
    seg_date = ['2016-01-01' , '2019-09-30']
    train_date = ['2016-01-01' , '2019-05-30']
    vald_date = ['2019-05-31', '2019-07-31']
    
    obj = Preprocess(path = file_path, seg_date = seg_date, train_date = train_date , vald_date = vald_date,
                     train_path = train_path, vald_path = vald_path)
    
    obj.data_split() 