import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sweetviz
from sklearn.metrics import confusion_matrix

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

df[['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week','income']] = MinMaxScaler().fit_transform(df[['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week','income']])

# my_report = sweetviz.analyze(df,target_feat='income')
# my_report.show_html()

x = df.drop(['income'],axis = 1)
y = df.income

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state= 6)

classifier = DecisionTreeClassifier()
classifier.fit(x_train,y_train)
print(classifier.score(x_test,y_test))
