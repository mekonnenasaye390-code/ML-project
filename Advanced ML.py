import pandas as pd
import matplotlib.pyplot as plt
import xgboost
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV



data=pd.read_csv(r'C:\Users\hp\OneDrive\AI projects\creditcard.csv')
print(data.head())
print(data.shape)
print(data.describe())
print(data.isnull().sum())
print(data.isnull())
print(data.info())
print(data['Class'].value_counts())

print(data['Class'].value_counts().plot(kind='bar'))
plt.scatter(data['Class'], data['Class'])
#plt.show()
## Feature scaling
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
scaler=StandardScaler()
scaler.fit_transform(data)
data['Amount'] = scaler.fit_transform(data[['Amount']])
data['Time'] = scaler.fit_transform(data[['Time']])
print('amount',data['Amount'])
print(data['Amount'].head())

x=data.drop('Class',axis=1)
y=data['Class']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=7)
print('x_train',x_train.shape)
print('y_train',y_train.shape)
print('x_train1',x_train)
print('y_train1',y_train)

## fix imbalance
from imblearn.over_sampling import SMOTE
sm=SMOTE(random_state=42)
sm.fit(x_train,y_train)
x_train_res, y_train_res=sm.fit_resample(x_train,y_train)
print('x_train_res',x_train_res.value_counts())
print('y_train_res',y_train_res.value_counts())


model=LogisticRegression()
model.fit(x_train,y_train)
y_pred1=model.predict(x_test)
print(accuracy_score(y_test,y_pred1))
print(precision_score(y_test,y_pred1))
print(recall_score(y_test,y_pred1))
print(f1_score(y_test,y_pred1))


## Bagging RandomForest
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
y_pred2=rf.predict(x_test)
print(accuracy_score(y_test,y_pred2))
print(precision_score(y_test,y_pred2))
print(recall_score(y_test,y_pred2))
print(f1_score(y_test,y_pred2))

## Boosting Gradient Bossting
from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier()
gb.fit(x_train,y_train)
y_pred3=gb.predict(x_test)
print(accuracy_score(y_test,y_pred3))
print(precision_score(y_test,y_pred3))
print(recall_score(y_test,y_pred3))
print(f1_score(y_test,y_pred3))

##SVM advanced classifier
from sklearn.svm import SVC
svc=SVC(kernel='rbf')
svc.fit(x_train,y_train)
y_pred4=svc.predict(x_test)
print(accuracy_score(y_test,y_pred4))
print(precision_score(y_test,y_pred4))
print(recall_score(y_test,y_pred4))
print(f1_score(y_test,y_pred4))

## Xgboost
from xgboost import XGBClassifier
xgb=XGBClassifier()
xgb.fit(x_train,y_train)
y_pred5=xgb.predict(x_test)
print(accuracy_score(y_test,y_pred5))
print(precision_score(y_test,y_pred5))
print(recall_score(y_test,y_pred5))
print(f1_score(y_test,y_pred5))

model = XGBClassifier()
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(x_train, y_train)


print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

estimator=XGBClassifier()
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2]
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

random_search.fit(x_train, y_train)

print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

## Final comparassion
models={
    'LogisticRegression':LogisticRegression(),
    'RandomForestClassifier':RandomForestClassifier(),
    'SVC':SVC(),
    'GradientBoostingClassifier':GradientBoostingClassifier(),
    'xgboost':xgboost,
}
for name,model in models.items():
    model.fit(x_train,y_train)
    pred=model.predict(x_test)
    print(name,accuracy_score(y_test,pred))
    print(precision_score(y_test,pred))
    print(recall_score(y_test,pred))
    print(f1_score(y_test,pred))

## Real Fraud prediction
def predict_fraud(model,sample):
    prediction=model.predict(sample)
    if prediction[0]==1:
        return "Fraud Detected_block transaction"
    else:
        return "Normal Transaction"





