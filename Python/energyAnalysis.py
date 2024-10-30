#%%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

#%%
df = pd.read_csv("/Users/stephenkullman/Desktop/Python/Projects/Building Energy Efficiency.csv")

#%%
df.isnull().sum()

#%%
df.hist(bins=20, figsize=(20,15))
plt.show()

#%%
df.corr()


#%%
Y1 = df[['Heating Load']]
Y2 = df[['Cooling Load']]

#%%
X = df[['Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area',
       'Overall Height', 'Orientation', 'Glazing Area',
       'Glazing Area Distribution']]
# %%
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from keras.layers import Dense
from keras.models import Sequential
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor,BaggingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score

#%%

X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X, Y1, Y2, test_size=0.33, random_state = 20)

MinMax = MinMaxScaler(feature_range= (0,1))
X_train = MinMax.fit_transform(X_train)
X_test = MinMax.transform(X_test)

#%%

Acc = pd.DataFrame(index=None, columns=['model','train_Heating','test_Heating','train_Cooling','test_Cooling'])

#%%
regressors = [['SVR',SVR()],
              
              ['DecisionTreeRegressor',DecisionTreeRegressor()],
              ['KNeighborsRegressor', KNeighborsRegressor()],
              ['RandomForestRegressor', RandomForestRegressor()],
              ['MLPRegressor',MLPRegressor()],
              ['AdaBoostRegressor',AdaBoostRegressor()],
              ['GradientBoostingRegressor',GradientBoostingRegressor()]]

#%%
for model in regressors:
    nameOfModel = model[0]
    model = model[1]

    model.fit(X_train, y1_train)
    r2_train_heating = r2_score(y1_train, model.predict(X_train))
    r2_test_heating = r2_score(y1_test, model.predict(X_test))


    model.fit(X_train, y2_train)
    r2_train_cooling = r2_score(y2_train, model.predict(X_train))
    r2_test2_cooling = r2_score(y2_test, model.predict(X_test))

    new_row = pd.DataFrame({'model': [nameOfModel], 'train_Heating': [r2_train_heating], 'test_Heating': [r2_test_heating], 
                            'train_Cooling': [r2_train_cooling], 'test_Cooling': [r2_test2_cooling]})
                             
    Acc = pd.concat([Acc, new_row], ignore_index=True) 

Acc.sort_values(by='test_Heating')

#%%
DTR = DecisionTreeRegressor()

paramGrid = {"criterion":["squared_error","absolute_error"], "min_samples_split":[14,16,18,20],
    "max_depth":[1,3,5],"min_samples_leaf":[4,5,6], "max_leaf_nodes": [29, 30, 31, 32],}

grid_search_CV = GridSearchCV(DTR, paramGrid, cv=5, refit=True)

grid_search_CV.fit(X_train, y2_train)

print("R-Squared::{}".format(grid_search_CV.best_score_))
print("Best Hyperparameters::\n{}".format(grid_search_CV.best_params_))


# %%
DTR_final = DecisionTreeRegressor(criterion= 'squared_error', max_depth= 5, max_leaf_nodes= 29, min_samples_leaf= 6, min_samples_split= 16)

DTR_final.fit(X_train,y1_train)
print("R-Squared on train dataset={}".format(DTR_final.score(X_test,y1_test)))

DTR_final.fit(X_train,y2_train)   
print("R-Squaredon test dataset={}".format(DTR_final.score(X_test,y2_test)))
# %%

y2_pred = DTR_final.predict(X_test)
# %%
DTR_final.fit(X_train,y1_train)
y1_pred = DTR_final.predict(X_test)

# %%
x_ax = range(len(y1_test))
plt.figure(figsize=(20,10))
plt.subplot(2,1,1)
plt.plot(x_ax, y1_test, label="Actual Heating")
plt.plot(x_ax, y1_pred, label="Predicted Heating")
plt.title("Heating test and predicted data")
plt.xlabel('X-axis')
plt.ylabel('Heating load (kW)')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(x_ax, y2_test, label="Actual Cooling")
plt.plot(x_ax, y2_pred, label="Predicted Cooling")
plt.title("Coolong test and predicted data")
plt.xlabel('X-axis')
plt.ylabel('Cooling load (kW)')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)

plt.show()
# %%
