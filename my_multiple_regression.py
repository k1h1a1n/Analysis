import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('50_Startups.csv')
 
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder=LabelEncoder()
X[:,3]=labelencoder.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[-1])
X=onehotencoder.fit_transform(X).toarray()

X=X[:,1:]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)

y_pred=reg.predict(X_test)

X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)

import statsmodels.regression.linear_model as sm
X_opt=X[:,[0,1,2,3,4,5]]
reg_OLS=sm.OLS(endog=y,exog=X_opt).fit()
reg_OLS.summary()

X_opt=X[:,[0,1,3,4,5]]
reg_OLS=sm.OLS(endog=y,exog=X_opt).fit()
reg_OLS.summary()

X_opt=X[:,[0,3,4,5]]
reg_OLS=sm.OLS(endog=y,exog=X_opt).fit()
reg_OLS.summary()

X_opt=X[:,[0,3,5]]
reg_OLS=sm.OLS(endog=y,exog=X_opt).fit()
reg_OLS.summary()

X_opt=X[:,[0,3]]
reg_OLS=sm.OLS(endog=y,exog=X_opt).fit()
reg_OLS.summary()

