import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,roc_auc_score
import sweetviz
from category_encoders.one_hot import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('adult.data')
print(df.head())
print(df.dtypes)
print(df.isnull().values.any())

columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']

for colum in df.columns:
    if df[colum].dtype == object:
        df[colum] = OneHotEncoder().fit_transform(df[colum])

df = MinMaxScaler().fit_transform(df)
df = pd.DataFrame(df, columns= columns)


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
# for i in np.arange(start = 0.06,stop = 0.20,step = 0.02):
#     features = get_features(i)
#     thresh.append((i))
#     X = df[features]
#     Y = df.income
#
#     x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=4)
#     classifier = LogisticRegression()
#     classifier.fit(x_train, y_train)
#     score = classifier.score(x_test, y_test)
#     scores.append(score)
# plt.plot(thresh,scores)
# plt.xlabel('thrshold_values')
# plt.ylabel('scores')
# plt.show()

features = get_features(0.13)
#print(features)

x = df[features]
y = df.income

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state= 4)

classifier = LogisticRegression()
classifier.fit(x_train,y_train)
print(classifier.score(x_test,y_test))

predictions = classifier.predict(x_test)
print(confusion_matrix(y_test,predictions))


probs = (classifier.predict_proba(x_test)[:,1])
#print(roc_auc_score(y_test,probs))

fpr, tpr, thresholds = roc_curve(y_test, probs)
# print(thresholds)

accuracy_ls = []
for thres in thresholds:
    y_pred = np.where(probs > thres, 1, 0)
    accuracy_ls.append(accuracy_score(y_test, y_pred, normalize=True))

accuracy_ls = pd.concat([pd.Series(thresholds), pd.Series(accuracy_ls)],
                        axis=1)
accuracy_ls.columns = ['thresholds', 'accuracy']
accuracy_ls.sort_values(by='accuracy', ascending=False, inplace=True)
# print(accuracy_ls.head())


threshold = accuracy_ls.iloc[1,0]
#print(threshold)
preds = np.where(classifier.predict_proba(x_test)[:,1] > threshold, 1, 0)
print(accuracy_score(y_test,preds))
