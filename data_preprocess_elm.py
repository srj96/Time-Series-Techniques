import pandas as pd 
import numpy as np 

class Preprocess:
    '''Below the data is split into train and test and then groups it according to
    time periods. The start date and end date are given in the main()'''
    
    def __init__(self,**kwargs):

        [self.__setattr__(k , kwargs[k]) for k in kwargs.keys()]
        
        
    def data_split(self):
        df = pd.read_csv(self.path)
        # df = df.rename(columns = {'mcp_price' : 'price'})
        df_1619 = df.loc[(df['date'] >= self.seg_date[0]) & (df['date'] <= self.seg_date[-1])]
        # df_1619= df_1619.sort_values(by = ['date'])
        # df_1619 = df_1619.reset_index()
        # print('df_1619 periods', sorted(df_1619['period'].unique()))
        # df_1619 = df_1619[df_1619['price'] != '']
        # df_1619['price'] = df_1619['price'].dropna()
        # df_1619 = df_1619.dropna(axis = 0)
        df_east= pd.DataFrame() 
        for ix in sorted(df_1619['period'].unique()):
            y = df_1619.loc[df_1619['period'] == ix]
            df_east = df_east.append(y)
            
            train_set = df_east.loc[(df_east['date'] >= self.train_date[0]) & (df_east['date'] <= self.train_date[-1])]
            test_set = df_east.loc[(df_east['date'] >= self.test_date[0]) & (df_east['date'] <= self.test_date[-1])]
            
            with open(self.train_path, 'w') as train_file:
                train_set.to_csv(train_file)
            with open(self.test_path, 'w') as test_file:
                test_set.to_csv(test_file) 

        return(test_set,train_set) 

if __name__ == '__main__':
    
    file_path = '/home/congnitensor/Python_projects/ptc_armax/ptc_area_data/features_labels_e1.csv'
    train_path = '/home/congnitensor/Python_projects/ptc_armax/train_test_elm/train_ptc_e1.csv'
    test_path = '/home/congnitensor/Python_projects/ptc_armax/train_test_elm/test_ptc_e1.csv'
    seg_date = ['2016-01-01' , '2019-09-30']
    train_date = ['2016-01-01' , '2019-07-30']
    test_date = ['2019-07-31', '2019-09-30']
    
    obj = Preprocess(path = file_path, seg_date = seg_date, train_date = train_date , test_date = test_date,
                     train_path = train_path, test_path = test_path)
    
    obj.data_split() 

