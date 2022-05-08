import pandas as pd
import numpy as np
from IPython.display import display, HTML
import numpy as np
import pandas as pd
from copy import copy, deepcopy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, KFold
from imblearn.over_sampling import RandomOverSampler
import smogn
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
import collections

from sklearn.feature_selection import SelectKBest, f_classif, f_regression, SelectFpr

models_B = [ # TARGET_B
    DecisionTreeClassifier(max_depth = 20), # tried 40, 60, 80, same
    LogisticRegression(max_iter = 100, solver = "liblinear"), # tried 200, 400, 800, same
    AdaBoostClassifier(n_estimators=100, learning_rate=0.5),
    MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, learning_rate_init=0.005),
    # # RandomForestClassifier(), # give very low profit
    # SVC(),
]
models_D = [
    # LinearRegression(),
    # MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, learning_rate_init=0.005),
    # SVR()
]

def compare_models(data, balanced_sampling=True):
    result = {
        m: {'acc': [], 'fp': [], 'fn': [], 'profit': []} for m in models_B + models_D
    }
    # train, test = train_test_split(data, test_size=0.3)
    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(data):
        # print("TRAIN:", train_index, "TEST:", test_index)
        tmp = deepcopy(data)
        train, test = tmp.iloc[train_index], tmp.iloc[test_index]
        # get targets out
        train_B = train["TARGET_B"]
        train_D = train["TARGET_D"]
        test_B = test["TARGET_B"]
        test_D = test["TARGET_D"]
        train.drop(columns = ["TARGET_D", "TARGET_B"], inplace = True)
        test.drop(columns = ["TARGET_D", "TARGET_B"], inplace = True)
        # we need to resample the train data to balance it out
        if balanced_sampling:
            x_res_B, y_res_B = RandomOverSampler(random_state=10000).fit_resample(train, train_B)
            train1 = deepcopy(train)
            train1['TARGET_D'] = train_D
            x_res_D, y_res_D = train, train_D # smogn.smoter(train1, 'TARGET_D', rel_method='manual') not working
        else:
            x_res_B, y_res_B = train, train_B
            x_res_D, y_res_D = train, train_D
        # print("oversampled to "+str(x_res_B.shape[0])+" data points for classification.")
        # run the model
        for clf in models_B + models_D:
            acc, fp, fn, profit = run_classifier(clf, x_res_B, y_res_B, test, test_B, test_D, regression=clf in models_D)
            result[clf]['acc'].append(acc)
            result[clf]['fp'].append(fp)
            result[clf]['fn'].append(fn)
            result[clf]['profit'].append(profit)
    for m in result:
        for metric in ['acc', 'fp', 'fn', 'profit']:
            print(m, np.mean(result[m][metric]))

def validate_models(train_data, val_data, balanced_sampling=True):
    result = {
        m: {'acc': None, 'fp': None, 'fn': None, 'profit': None} for m in models_B + models_D
    }
    tmp = deepcopy(data)
    train, test = train_data, val_data
    # get targets out
    train_B = train["TARGET_B"]
    train_D = train["TARGET_D"]
    test_B = test["TARGET_B"]
    test_D = test["TARGET_D"]
    train.drop(columns = ["TARGET_D", "TARGET_B"], inplace = True)
    test.drop(columns = ["TARGET_D", "TARGET_B"], inplace = True)
    # we need to resample the train data to balance it out
    if balanced_sampling:
        x_res_B, y_res_B = RandomOverSampler(random_state=10000).fit_resample(train, train_B)
        train1 = deepcopy(train)
        train1['TARGET_D'] = train_D
        x_res_D, y_res_D = train, train_D # smogn.smoter(train1, 'TARGET_D', rel_method='manual') not working
    else:
        x_res_B, y_res_B = train, train_B
        x_res_D, y_res_D = train, train_D
    # print("oversampled to "+str(x_res_B.shape[0])+" data points for classification.")
    # run the model
    for clf in models_B + models_D:
        acc, fp, fn, profit = run_classifier(clf, x_res_B, y_res_B, test, test_B, test_D, regression=clf in models_D)
        result[clf]['acc'] = acc
        result[clf]['fp'] = fp
        result[clf]['fn'] = fn
        result[clf]['profit'] = profit
    for m in result:
        for metric in ['acc', 'fp', 'fn', 'profit']:
            print(m, result[m][metric])

def run_classifier(clf, x_res, y_res, test, test_B, test_D, regression=False):
    print(clf)
    clf = clf.fit(x_res, y_res)
    y_pred = clf.predict(test)
    if regression:
        acc, fp, fn, profit = get_acc_regressor(y_pred, test_B, test_D, 0.68)
    else:
        acc, fp, fn, profit = get_acc(y_pred, test_B, test_D, 0.68)
    print("accuracy = "+str(acc))
    print("false positive rate = "+str(fp))
    print("false negative rate = "+str(fn))    
    print("profit = "+str(profit))
    return acc, fp, fn, profit
    
def get_acc(y_pred, y_actual, y_donate, mail_cost):
    df = pd.concat([pd.Series(y_pred, index=y_actual.index), pd.Series(y_actual), pd.Series(y_donate)], axis = 1)
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

def get_acc_regressor(y_pred, y_actual, y_donate, mail_cost):
    df = pd.concat([pd.Series(y_pred), pd.Series(y_actual), pd.Series(y_donate)], axis = 1)
    df.columns = ["y_pred", "y_actual", "y_donate"]
    
    #get accuracy
    accuracy = df[((df['y_pred'] > mail_cost) & (df['y_actual'])) | ((df['y_pred'] <= mail_cost) & (df['y_actual'] == 0))].shape[0] / y_actual.shape[0]
    # get false positive rate
    fp_rate = df[(df['y_pred'] > mail_cost) & (df['y_actual'] == 0)].shape[0] / y_actual.shape[0]
    # get false negative rate
    fn_rate = df[(df['y_pred'] <= mail_cost) & (df['y_actual'] == 1)].shape[0] / y_actual.shape[0]
    # get total profit 
    profit = df[(df['y_pred'] > mail_cost) & (df['y_actual'] == 1)]['y_donate'].sum() - df[(df['y_pred'] > mail_cost)].shape[0]*mail_cost
    
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
    
    df.drop(labels=['CONTROLN', 'ZIP'], axis = 1, inplace=True)
    df.select_dtypes(include=['object']).replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df.select_dtypes(include=np.number).replace(r'^\s.*$', np.nan, regex=True, inplace=True)
    
    
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
    for key in df.select_dtypes(include=['object']).columns:
        mapping = {k: v+1 for v, k in enumerate(sorted([str(a) for a  in df[key].unique()]))} 
        df[key].replace(mapping, inplace=True)
    ####Time Frame and Date Fields#########
    end_date = 9706
    for time_key in ['MAXADATE', 'MINRDATE', 'MAXRDATE', 'LASTDATE', 'FISTDATE', 'NEXTDATE', 'ODATEDW']: 
        end_date = pd.to_datetime(end_date, format='%y%m', exact=True)
        df.loc[df[time_key] == 0, time_key] = df[time_key].mode()
        start_date = temp_date_attr = pd.to_datetime(df[time_key], format='%y%m', exact=True)
        df.loc[:,time_key] = (end_date - start_date).dt.days/30
    df.fillna(df.mode().iloc[0], inplace=True)
    ####Fields Containing Constants################
    df = df.loc[:, (df != df.iloc[0]).any()] 
    return df

## cross-validation on training set

# df = pd.read_csv("cup98lrn.txt", sep=',', error_bad_lines = False, low_memory = False, skip_blank_lines = True)
# data_trimmed = preprocessing_data(df)
# data_trimmed.to_csv('data_trimmed.csv', index = False)

data_trimmed = pd.read_csv("data_trimmed.csv", sep=',', error_bad_lines = False, low_memory = False, skip_blank_lines = True)
targets = deepcopy(data_trimmed[['TARGET_D', 'TARGET_B']])
data_trimmed.drop(columns = ['TARGET_D', 'TARGET_B'], inplace = True)
data_trimmed = (data_trimmed - data_trimmed.min())/(data_trimmed.max() - data_trimmed.min())

features = SelectKBest(score_func=f_classif, k=200).fit_transform(data_trimmed, targets["TARGET_B"]) # f_regression for "TARGET_D"
data_selected = pd.DataFrame(features)
data = pd.concat([data_selected, targets], axis = 1)
# compare_models(data, balanced_sampling=False)

## test on validation set

# df1 = pd.read_csv("cup98val.txt", sep=',', error_bad_lines = False, low_memory = False, skip_blank_lines = True)
# df1_gt = pd.read_csv("valtargt.txt", sep=',', error_bad_lines = False, low_memory = False, skip_blank_lines = True)
# data_trimmed_val = preprocessing_data(pd.merge(df1, df1_gt, on='CONTROLN'))
# data_trimmed_val.to_csv('data_trimmed_val.csv', index = False)

data_trimmed_val = pd.read_csv("data_trimmed_val.csv", sep=',', error_bad_lines = False, low_memory = False, skip_blank_lines = True)
targets_val = deepcopy(data_trimmed_val[['TARGET_D', 'TARGET_B']])
data_trimmed_val.drop(columns = ['TARGET_D', 'TARGET_B'], inplace = True)
data_trimmed_val = (data_trimmed_val - data_trimmed_val.min())/(data_trimmed_val.max() - data_trimmed_val.min())

features_val = SelectKBest(score_func=f_classif, k=200).fit_transform(data_trimmed_val, targets_val["TARGET_B"]) # f_regression for "TARGET_D"
data_selected_val = pd.DataFrame(features_val)
data_val = pd.concat([data_selected_val, targets_val], axis = 1)

validate_models(data, data_val, balanced_sampling=True)