
import ROOT
import uproot # can also use root_pandas or root_numpy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, auc
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
#%matplotlib inline
import time
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-d","--dir", dest="dir", type='string',default='./',
                  help="Specify the working directory")
(options, args) = parser.parse_args()

directory = options.dir

print("working")

#with open ("df_3prong_Run2UL.pkl",'r') as f:
#    X=pickle.load(f)

print('opening df from csv file')
X = pd.read_csv(directory+'/df_3prong_Run2UL.csv')

print('finished opening df from csv file')

y = X["tauFlag"]
y.columns = ["class"]
X = X.drop("tauFlag", axis=1)


print('splitting into test and train datasets')
X_train, X_test, y_train, y_test  = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=123452,
    stratify=y.values,
)
print('done splitting into test and train datasets')
del X, y

print 'variables used in training: '
print X_train.columns.values


xgb_params = {
    "objective": "multi:softprob",
    "max_depth": 5,
    "learning_rate": 0.05,
    "silent": 1,
    "n_estimators": 1000,
    "subsample": 0.9,
    "seed": 123451,
    "num_class": 2,
    'tree_method': 'gpu_hist',
    #'predictor': 'cpu_predictor' # prevents excessive memory usage but slower
}


print ('training model')
start_time = time.time()
xgb_clf = xgb.XGBClassifier(**xgb_params)
xgb_clf.fit(
    X_train,
    y_train,
    early_stopping_rounds=10,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric = "mlogloss",
    verbose=True,
)
elapsed_time = time.time() - start_time
print ('finished training model, elapsed time=', elapsed_time)


print ('saving model')
xgb_clf.get_booster().save_model('mvadm_3prong_Run2UL.model')


with open ("mvadm_3prong_Run2UL.pkl",'w') as f:
    pickle.dump(xgb_clf,f)
print 'model saved!'

proba_train=xgb_clf.predict_proba(X_train)
proba_test=xgb_clf.predict_proba(X_test)

with open ("mvadm_3prong_proba_train.pkl",'w') as f:
    pickle.dump(proba_train,f)

with open ("mvadm_3prong_proba_test.pkl",'w') as f:
    pickle.dump(proba_test,f)

with open ("mvadm_3prong_X_train.pkl",'w') as f:
    pickle.dump(X_train,f)

with open ("mvadm_3prong_y_odd.pkl",'w') as f:
    pickle.dump(y_train,f)

with open ("mvadm_3prong_X_test.pkl",'w') as f:
    pickle.dump(X_test,f)

with open ("mvadm_3prong_y_test.pkl",'w') as f:
    pickle.dump(y_test,f)

