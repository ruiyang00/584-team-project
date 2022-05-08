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

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
def compare_models(data):
    # train test split
    train, test = train_test_split(data, test_size=0.3)
    # get targets out
    train_B = train["TARGET_B"]
    train_D = train["TARGET_D"]
    test_B = test["TARGET_B"]
    test_D = test["TARGET_D"]
    train.drop(columns = ["TARGET_D","TARGET_B"], inplace = True)
    test.drop(columns = ["TARGET_D","TARGET_B"], inplace = True)
    # we need to resample the train data to balance it out
    sampler = RandomOverSampler(random_state=50)
    x_res, y_res = sampler.fit_resample(train, train_B)
    print("oversampled to "+str(x_res.shape[0])+"data points.")
    
    # if dimension_red_model is used, use it on test
    #if(dimension_red_model != None):
    #    test = dimension_red_model.fit_transform(test)
    
    # run the model
    acc_log, fp_rate_log, fn_rate_log, profit_log = run_regression(x_res, y_res, test, test_B, test_D)
    print("logistic regression accuracy = "+str(acc_log))
    print("logistic regression false positive rate = "+str(fp_rate_log))
    print("logistic regression false negative rate = "+str(fn_rate_log))    
    print("logistic regression profit = "+str(profit_log))
    
    # run decision tree
    acc_tree, fp_rate_tree, fn_rate_tree, profit_tree = run_decision_tree(x_res, y_res, test, test_B, test_D)
    print("decision tree accuracy = "+str(acc_tree))
    print("decision tree false positive rate = "+str(fp_rate_tree))
    print("decision tree false negative rate = "+str(fn_rate_tree))    
    print("decision tree profit = "+str(profit_tree))
def run_regression(x_res, y_res, test, test_B, test_D):
    # train the model
    clf = DecisionTreeClassifier(max_depth = 20)
    clf = clf.fit(x_res, y_res)
    
    # test on the test set
    y_pred = clf.predict(test)
    
    return get_acc(y_pred, test_B, test_D, 0.68)
def run_decision_tree(x_res, y_res, test, test_B, test_D):
    # train the model
    clf = LogisticRegression(max_iter = 100, solver = "liblinear", verbose = 1)
    clf = clf.fit(x_res, y_res)
    
    # test on the test set
    y_pred = clf.predict(test)
    
    return get_acc(y_pred, test_B, test_D, 0.68)
def get_acc(y_pred, y_actual, y_donate, mail_cost):
    df = pd.concat([pd.Series(y_pred), pd.Series(y_actual), pd.Series(y_donate)], axis = 1)
    df.columns = ["y_pred", "y_actual", "y_donate"]
    
    #get accuracy
    accuracy = df[(df['y_pred'] == df['y_actual'])].shape[0] / y_actual.shape[0]
    # get false positive rate
    fp_rate = df[(df['y_pred'] == 1) & (df['y_actual'] == 0)].shape[0] / y_actual.shape[0]
    # get false negative rate
    fn_rate = df[(df['y_pred'] == 0) & (df['y_actual'] == 1)].shape[0] / y_actual.shape[0]
    # get total profit 
    profit = df[(df['y_pred'] == 1) & (df['y_actual'] == 1)]["y_donate"].sum() - df[(df['y_pred'] == 1)].shape[0]*mail_cost
    
    return accuracy, fp_rate, fn_rate, profit
def preprocessing_data(df):     
    # need to be done first
    for key in ['NOEXCH', 'RECINHSE', 'RECP3', 'RECPGVG', 'RECSWEEP', 'MAILCODE', 'PEPSTRFL']:
        df.loc[df[key].isin(["0", "1", " ", 0, 1]), key]= 0
        df.loc[df[key].isin(["X"]), key] = 1
    
    df.loc[:,'ZIP'] = df.loc[:,'ZIP'].astype(str)
    df.loc[:,'ZIP'] = df.loc[:,'ZIP'].str.slice(0,5)
    
    
    ''' General:
        replacing any value with period or/and whitespace
    '''
    
    #whitesapce \s
    
    df.drop(labels=['CONTROLN'], axis = 1)
    df.select_dtypes(include=['object']).replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df.select_dtypes(include=np.number).replace(r'^\s.*$', np.nan, regex=True, inplace=True)
    
    
    ####dealing with missing features#################   
    #1. drop the attribute if missing values >= 99.5%
    #calculating the dropping_treshold 
    num_rows = len(df)
    perc = 98
    min_count =  int(((100-perc)/100)*num_rows+ 1)
    df.dropna(axis = 1, thresh=min_count)
    
    #2. if features contains NAN < 99.5% we need to replace NAN with the most frequent value
    #this line does replace differnet attribute types(Number, char, boolean, etc)  with the most frequent
    # value
    df.fillna(df.mode().iloc[0], inplace=True)
    
    
    ### categorical data ##########
    for key in df.select_dtypes(include=['object']).columns:
        mapping = {k: v for v, k in enumerate(df[key].unique())} 
        df[key].replace(mapping, inplace=True)
    ####Time Frame and Date Fields#########
    end_date = 9706
    for time_key in ['MAXADATE', 'MINRDATE', 'MAXRDATE', 'LASTDATE', 'FISTDATE', 'NEXTDATE', 'ODATEDW']: 
        end_date = pd.to_datetime(end_date, format='%y%m', exact=True)
        df.loc[df[time_key] == 0, time_key] = df[time_key].mode()
        start_date = temp_date_attr = pd.to_datetime(df[time_key], format='%y%m', exact=True)
        df.loc[:,time_key] = (end_date - start_date).dt.days
    df.fillna(df.mode().iloc[0], inplace=True)
    ####Fields Containing Constants################
    df.dropna(axis=1, thresh= 2, inplace=True)

    return df
def get_uniques(df):
    for col in df.columns:
        print(col + ':', df[col].unique())   
def pca_compress(data, var=0.95):
    # get pca
    pca_dims = PCA()
    pca_dims.fit(data)
    cumsum = np.cumsum(pca_dims.explained_variance_ratio_)
    d = np.argmax(cumsum >= var) + 1
    pca = PCA(n_components=d)
    output = pca.fit_transform(data)
    return output, pca

df = pd.read_csv("cup98lrn.txt", sep=',', error_bad_lines = False, low_memory = False, skip_blank_lines = True)
data_trimmed = preprocessing_data(df)
data_trimmed.to_csv('data_trimmed.csv', index = False)
# data_trimmed = pd.read_csv("data_trimmed.csv", sep=',', error_bad_lines = False, low_memory = False, skip_blank_lines = True)
targets = deepcopy(data_trimmed[['TARGET_D', 'TARGET_B']])

test = SelectKBest(score_func=f_classif, k=100)
fit = test.fit(data_trimmed, targets["TARGET_B"])
# summarize scores

print(fit.scores_)
features = fit.transform(data_trimmed)

data_selected = pd.DataFrame(features)

data = pd.concat([data_selected,targets], axis = 1)
compare_models(data)