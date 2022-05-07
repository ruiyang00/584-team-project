#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from IPython.display import display, HTML
import numpy as np
import pandas as pd
from copy import copy, deepcopy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import collections


# In[2]:


def compare_models(data, dimension_red_model):
    # train test split
    train, test = train_test_split(data, test_size=0.3)
    # get targets out
    train_B = train["TARGET_B"]
    train_D = train["TARGET_D"]
    test_B = test["TARGET_B"]
    test_D = test["TARGET_D"]
    train.drop(columns = ["TARGET_D","TARGET_B"], inplace = True)
    
    # we need to resample the train data to balance it out
    sampler = RandomOverSampler(random_state=50)
    x_res, y_res = over_sampler.fit_resample(train, train_B)
    print("oversampled to "+str(x_res.shape[0])+"data points.")
    
    # if dimension_red_model is used, use it on test
    if(dimension_red_model != None):
        test = dimension_red_model.fit_transform(test)
    
    # run the model
    acc_log, fp_rate_log, fn_rate_log, profit_log = run_regression(x_res, y_res, test)
    print("logistic regression accuracy = "+str(acc_log))
    print("logistic regression false positive rate = "+str(fp_rate_log))
    print("logistic regression false negative rate = "+str(fn_rate_log))    
    print("logistic regression profit = "+str(profit_log))
    
    # run decision tree
    acc_tree, fp_rate_tree, fn_rate_tree, profit_tree = run_decision_tree(x_res, y_res, test)
    print("decision tree accuracy = "+str(acc_tree))
    print("decision tree false positive rate = "+str(fp_rate_tree))
    print("decision tree false negative rate = "+str(fn_rate_tree))    
    print("decision tree profit = "+str(profit_tree))


# In[3]:


def run_regression(x_res, y_res, test):
    # train the model
    clf = DecisionTreeClassifier(max_depth = 20)
    clf = clf.fit(x_res, y_res)
    
    # test on the test set
    y_pred = clf.predict(test)
    
    return get_acc(y_pred, test_B, test_D, 0.68)


# In[4]:


def run_decision_tree(x_res, y_res, test):
    # train the model
    clf = LogisticRegression(max_iter = 100, solver = "liblinear", verbose = 1)
    clf = clf.fit(x_res, y_res)
    
    # test on the test set
    y_pred = clf.predict(test)
    
    return get_acc(y_pred, test_B, test_D, 0.68)


# In[5]:


def get_acc(y_pred, y_actual, y_donate, mail_cost):
    df = pd.concat([y_pred, y_actual, y_donate], axis = 1)
    df.columns = ["y_pred", "y_actual", "y_donate"]
    
    #get accuracy
    accuracy = df[(df['y_pred'] == df['y_actual'])].shape[0]
    # get false positive rate
    fp_rate = df[(df['y_pred'] == 1) & (df['y_actual'] == 0)].shape[0]
    # get false negative rate
    fn_rate = df[(df['y_pred'] == 0) & (df['y_actual'] == 1)].shape[0]
    # get total profit 
    profit = df[(df['y_pred'] == 1) & (df['y_actual'] == 1)]["y_donate"].sum() - df[(df['y_pred'] == 1)].shape[0]*mail_cost
    
    return accuracy, fp_rate, fn_rate, profit


# In[25]:


def preprocessing_data(df): 
    def split_types(in_dict):
        res = collections.defaultdict(list)
        for key, val in in_dict.items():
            if key in selected_features:
                res[val].append(key)
        return res

    def clean_whitespace_in_features(df):
        for column in df.columns:
            uniques = df[column].unique()
            if len(uniques) == 2 and ' ' in uniques:
                if is_string_dtype(df[column]):
                    df.loc[:,column] = df[:,column].replace(to_replace=' ', value='0')
                else:
                    df[:,column] = df[:,column].replace(to_replace=' ', value=0)    
            
            if len(uniques) > 2 and ' ' in uniques:
                df[:,column] = df[:,column].replace(to_replace=' ', value=np.nan)

    def get_df_data_types(df):
        for column in df.columns:
            print(column, df[column].unique())

    def get_wrong_types(df, in_dict):
        res = collections.defaultdict(list)
        for column in df.columns:
            if column in in_dict['Num'] and not is_numeric_dtype(df[column]):
                res['Num'] = column
            
            if column in in_dict['Char'] and not is_string_dtype(df[column]):
                res['Char'] = column
        return res;

    def replace_string_type_missing_values(string_types, df):
        for col in string_types:
            counts = len(df[col].unique())
            if counts == 2:
                df.loc[:,col] = df.loc[:,col].replace(to_replace=' ', value='Q')
            elif counts > 2:
                df.loc[:,col] = df.loc[:,col].replace(to_replace=' ', value=np.nan)         
            
    def replace_numeric_type_missing_values(num_types, df):
        for col in num_types:
            counts = len(df[col].unique())
            if counts == 2:
                df.loc[:,col] = df.loc[:,col].replace(r'\s.*$', value=9, regex=True)
            elif counts > 2:
                df.loc[:,col] = df.loc[:,col].replace(r'\s.*$', value=np.nan, regex=True)
            
    def perform_one_hot_encoding(list_attributes, df):
        for attr in list_attributes:
            pd.get_dummies(df[attr], prefix=attr)   
    ##some features had to manual preprocess#####
    ###truncate the ZIP atrribute to length=5####
    
    # need to be done first
    for key in ['NOEXCH', 'RECINHSE', 'RECP3', 'RECPGVG', 'RECSWEEP']:
        df.loc[df[key].isin(["0", "1", " ", 0, 1]), key]= 0
        df.loc[df[key].isin(["X"]), key] = 1
    
    type_dict = pd.read_csv('otype1.txt', header=None, index_col=0, squeeze=True).to_dict()
    type_list = split_types(type_dict)   
    
    df.loc[:,'RFA_2F'] = df.loc[:,'RFA_2F'].astype(str)
    df.loc[:,'ZIP'] = df.loc[:,'ZIP'].astype(str)
    df.loc[:,'ZIP'] = df.loc[:,'ZIP'].str.slice(0,5)
    
    ''' General:
        replacing any value with period or/and whitespace
    '''
    
    #whitesapce \s
    replace_string_type_missing_values(type_list['Char'], df)
    replace_numeric_type_missing_values(type_list['Num'], df)
    
    # df.drop(labels=['CONTROLN'], axis = 1)
    
    ####Fields Containing Constants################
    df.dropna(axis=1, thresh= 2, inplace=True)
    
    ####dealing with missing features#################   
    #1. drop the attribute if missing values >= 99.5%
    #calculating the dropping_treshold 
    num_rows = len(df)
    perc = 99.5
    min_count =  int(((100-perc)/100)*num_rows+ 1)
    df.dropna(axis = 1, thresh=min_count)
    
    #2. if features contains NAN < 99.5% we need to replace NAN with the most frequent value
    #this line does replace differnet attribute types(Number, char, boolean, etc)  with the most frequent
    # value
    df.fillna(df.mode().iloc[0], inplace=True)
    
    ### categorical data ##########
    for key in type_list['Char']:
        mapping = {k: v for v, k in enumerate(df[key].unique())} 
        df[key].replace(mapping, inplace=True)
    ####Time Frame and Date Fields#########
    end_date = 9706
    for time_key in ['MAXADATE', 'MINRDATE', 'MAXRDATE', 'LASTDATE', 'FISTDATE', 'NEXTDATE', 'ODATEDW']: # 
        end_date = pd.to_datetime(end_date, format='%y%m', exact=True)
        df.loc[df[time_key] == 0, time_key] = df[time_key].mode()
        start_date = temp_date_attr = pd.to_datetime(df[time_key], format='%y%m', exact=True)
        df.loc[:,time_key] = (end_date - start_date).dt.days
    df.fillna(df.mode().iloc[0], inplace=True)
    
    return df


# In[7]:


def pca_compress(data, var=0.95):
    # get pca
    pca_dims = PCA()
    pca_dims.fit(data)
    cumsum = np.cumsum(pca_dims.explained_variance_ratio_)
    d = np.argmax(cumsum >= var) + 1
    pca = PCA(n_components=d)
    output = pca.fit_transform(data)
    print("pca done")
    return output, pca


# In[26]:


pd.get_option("display.max_columns")
df = pd.read_csv("cup98lrn.txt", sep=',', error_bad_lines = False, low_memory = False, skip_blank_lines = True)

selected_features = [
    'TARGET_D', 'TARGET_B',
    'OSOURCE', 'ZIP', 'PVASTATE', 'DOB', 'NOEXCH', 'RECP3', 'RECINHSE', 'RECPGVG', 'RECSWEEP', 'DOMAIN', 'CLUSTER', 'AGE', 'HOMEOWNR', 'NUMCHLD', 'INCOME', 'GENDER', 'WEALTH1', 'HIT', 'PUBNEWFN', 'MALEMILI', 'MALEVET', 'VIETVETS', 'WWIIVETS', 'LOCALGOV', 'STATEGOV', 'FEDGOV', 'WEALTH2', 'COLLECT1', 'VETERANS', 'BIBLE', 'CATLG', 'HOMEE', 'PETS', 'CDPLAY', 'STEREO', 'PCOWNERS', 'PHOTO', 'CRAFTS', 'FISHER', 'GARDENIN', 'BOATS', 'WALKER', 'KIDSTUFF', 'CARDS', 'PLATES', 'LIFESRC', 'ETH3', 'ETH8', 'ETH10', 'ETH11', 'ETH15', 'OEDC1', 'OEDC2', 'OEDC3', 'CARDPROM', 'MAXADATE', 'NUMPROM', 'CARDPM12', 'NUMPRM12', 'RAMNTALL', 'NGIFTALL', 'CARDGIFT', 'MINRAMNT', 'MINRDATE', 'MAXRAMNT', 'MAXRDATE', 'LASTGIFT', 'LASTDATE', 'FISTDATE', 'NEXTDATE', 
    'TIMELAG', 'AVGGIFT', 'RFA_2R', 'RFA_2F', 'RFA_2A', 'MDMAUD_R', 'MDMAUD_F', 'MDMAUD_A', 'CLUSTER2', 'GEOCODE2', 'ODATEDW']

# print(df[selected_features])
# df[selected_features].to_csv('selected_features.csv', index=False)

data_trimmed = preprocessing_data(df[selected_features])
targets = deepcopy(data_trimmed[['TARGET_D', 'TARGET_B']])
data_trimmed.drop(columns = ["TARGET_D","TARGET_B"], inplace = True)
#pd.get_dummies(data_trimmed).shape
data, pca_model = pca_compress(pd.get_dummies(data_trimmed))
data = pd.concat([data,targets], axis = 1)
compare_models(data)

