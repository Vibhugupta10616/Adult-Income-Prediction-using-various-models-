import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
import sweetviz
from category_encoders.one_hot import OneHotEncoder

df = pd.read_csv('adult.data',header = 0, names =['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income'])
# print(df.head())
# print(df.dtypes)
# print(df.isnull().values.any())


for colum in df.columns:
    if df[colum].dtype == object:
        df[colum] = OneHotEncoder().fit_transform(df[colum])

df = MinMaxScaler().fit_transform(df)
df = pd.DataFrame(df, columns=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income'])


# my_report = sweetviz.analyze(df,target_feat='income')
# my_report.show_html()
#
correlations = df.corr()['income'].drop('income')
print(correlations)
print(correlations.quantile(.25))
print(correlations.quantile(.75))
print(correlations.quantile(.50))
# sns.heatmap(df.corr(), fmt = '.2f',annot = True)
# plt.show()

def get_features(correlation_threshold):
    abs_corrs = correlations.abs()
    high_correlations = abs_corrs[abs_corrs > correlation_threshold].index.values.tolist()
    return high_correlations

# thresh = []
# scores = []
# for i in np.arange(start = 0.10,stop = 0.25,step = 0.02):
#     features = get_features(i)
#     thresh.append((i))
#     X = df[features]
#     Y = df.income
#
#     x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=4)
#     classifier = SVC()
#     classifier.fit(x_train, y_train)
#     score = classifier.score(x_test, y_test)
#     scores.append(score)
# plt.plot(thresh,scores)
# plt.xlabel('thrshold_values')
# plt.ylabel('scores')
# plt.show()

features = get_features(0.22)

x = df[features]
y = df.income

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state= 6)


classifier = SVC(kernel = 'rbf',gamma = 0.5,C = 10,random_state=8)
classifier.fit(x_train,np.ravel(y_train))
print(classifier.score(x_test,np.ravel(y_test)))

predictions = classifier.predict(x_test)
print(confusion_matrix(y_test,predictions))
#
# parameters = [{'C':[1,10,1000] ,'kernel':['linear']},
#               {'C':[1,10,100], 'kernel' : ['rbf'],'gamma' :[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]}]
#
# search = RandomizedSearchCV(estimator=classifier,param_distributions= parameters,scoring='accuracy',cv = 3)
# search.fit(x_train,np.ravel(y_train))
#
# print(search.best_score_)
# print(search.best_params_)
