import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
import sweetviz

df = pd.read_csv('adult.data')
#print(df.head())
#print(df.isnull().values.any())

df.columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']

def mapping_function(df_column):
    dic = {}
    b = 0
    for a in df_column.unique():
        dic[a] = b
        b += 1
    return (dic)

for colum in df.columns:
    if df[colum].dtype == object:
        df[colum] = df[colum].map(mapping_function(df[colum]))


df[['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']] = MinMaxScaler().fit_transform(df[['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']])

# my_report = sweetviz.analyze(df,target_feat='income')
# my_report.show_html()

x = df.drop(['income'],axis = 1)
y = df.income

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state= 6)


classifier = SVC()
classifier.fit(x_train,np.ravel(y_train))
print(classifier.score(x_test,np.ravel(y_test)))

predictions = classifier.predict(x_test)
print(confusion_matrix(y_test,predictions))
#
# parameters = [{'C':[1,10,1000] ,'kernel':['linear']},
#               {'C':[1,10,100], 'kernel' : ['rbf'],'gamma' :[0.1,0.2,0.3,0.4,0.5]}]
#
# search = RandomizedSearchCV(estimator=classifier,param_distributions= parameters,scoring='accuracy',cv = 3)
# search.fit(x_train,np.ravel(y_train))
#
# print(search.best_score_)
# print(search.best_params_)