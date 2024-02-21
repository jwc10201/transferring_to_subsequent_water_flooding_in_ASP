# -*- encoding: utf-8 -*-
from autosklearn.estimators import AutoSklearnRegressor
import pandas as pd
import pickle

###############Read data################
autoskdata=pd.read_excel('15_819.xlsx',index_col=None)


x=autoskdata.drop("y", axis=1)
y=autoskdata["y"]



################Dataset partitioning################
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,random_state=50,test_size=0.20)



##############Modeling(train 24hours)#############
automl = AutoSklearnRegressor(time_left_for_this_task=82800, per_run_time_limit=600,ml_memory_limit=61440, ensemble_memory_limit=10240)



'''
##############Modeling(train 5mins)#############
automl_1 = AutoSklearnRegressor(time_left_for_this_task=420, per_run_time_limit=40,ml_memory_limit=61440, ensemble_memory_limit=10240)
'''



##########Trainning##########
automl.fit(x_train, y_train)


##########Generate pickle file##########
with open('automl_ASP.pickle','wb') as fw_1:
    pickle.dump(automl,fw_1)

