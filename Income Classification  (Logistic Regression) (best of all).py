import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
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

df[['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']] = StandardScaler().fit_transform(df[['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']])

# my_report = sweetviz.analyze(df,target_feat='income')
# my_report.show_html()

correlations = df.corr()['income'].drop('income')
# print(correlations)
#
# sns.heatmap(df.corr(),fmt = '.2f',annot = True)
# plt.show()

def get_features(correlation_threshold):
    abs_corrs = correlations.abs()
    high_correlations = abs_corrs[abs_corrs > correlation_threshold].index.values.tolist()
    return high_correlations

features = get_features(0.1)
#print(features)

x = df[features]
y = df.income
#print(y.unique())

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state= 6)

classifier = LogisticRegression()
classifier.fit(x_train,y_train)
# print(classifier.score(x_test,y_test))

#predictions = classifier.predict(x_test)
#print(confusion_matrix(y_test,predictions))


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