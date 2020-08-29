import pandas as pd
df=pd.read_csv(r'C:/Users/SAIDHANUSH/ipl_modify.csv')
df.head()

X=df.iloc[:,[2,3,4,5,6,7,8]]
y=df.iloc[:,[9]]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33)

from sklearn.linear_model import LinearRegression
model=LinearRegression()

model.fit(X_train,y_train)

prediction=model.predict(X_test)

import pickle

pickle.dump(model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))