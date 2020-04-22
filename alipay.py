from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd

data = pd.read_csv('alipay_huabei_GS1.csv')
n = data.shape[1] - 1
x = data.iloc[:, 1:n]
y = data.iloc[:, -1]
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.30, stratify=y, random_state=3000)

# 不需要Max_Min的分类器
classifiers1 = [KNeighborsClassifier(),
                SVC(),
                DecisionTreeClassifier(),
                RandomForestClassifier(),
                AdaBoostClassifier()]

# 分类器名称
classifiers_names1 = ['kneighborsClassifier',
                      'svc',
                      'decisionTreeClassifier',
                      'randomForestClassifier',
                      'adaBoostClassifier']

# 分类器参数
classifiers_param_grid1 = [{'kneighborsClassifier__n_neighbors': range(3, 11)},
                           {'svc__C': [1, 0.5, 1.5]},
                           {'decisionTreeClassifier__max_depth': range(3, 11)},
                           {'randomForestClassifier__n_estimators': range(3, 11)},
                           {'adaBoostClassifier__n_estimators': [10, 20, 50, 100]}]

def pca_classifier(classifiers1, classifiers_names1, classifiers_param_grid1, train_x, train_y, test_x):
    for classifier, classifier_name, classifier_param_grid in zip(classifiers1, classifiers_names1, classifiers_param_grid1):
        pipeline = Pipeline([('ss', StandardScaler()),
                             ('pca', PCA(n_components=2)),
                             (classifier_name, classifier)])
        gridsearch = GridSearchCV(estimator=pipeline, param_grid=classifier_param_grid, scoring='accuracy')
        search = gridsearch.fit(train_x, train_y)
        predict_y = gridsearch.predict(test_x)
        print('GridSearch 最优参数：', search.best_params_)
        print('GridSearch 最优分数：%0.4lf' % search.best_score_)
        print('准确率: %0.4lf' % accuracy_score(test_y, predict_y))

# 需要Max_Min的分类器
classifiers2 = [MultinomialNB(),
                LogisticRegression()]

# 分类器名称
classifiers_names2 = ['MultinomialNB',
                      'LogisticRegression']

# 分类器参数
classifiers_param_grid2 = [{'MultinomialNB__alpha': [1, 0.5, 1.5]},
                          {'LogisticRegression__penalty': ['l2'], 'LogisticRegression__max_iter': [1000]}]

def mm_classifier(classifiers2, classifiers_names2, classifiers_param_grid2, train_x, train_y, test_x):
    for classifier, classifier_name, classifier_param_grid in zip(classifiers2, classifiers_names2, classifiers_param_grid2):
        pipeline = Pipeline([('ss', StandardScaler()),
                             ('mm', MinMaxScaler()),
                             (classifier_name, classifier)])
        gridsearch = GridSearchCV(estimator=pipeline, param_grid=classifier_param_grid, scoring='accuracy')
        search = gridsearch.fit(train_x, train_y)
        predict_y = gridsearch.predict(test_x)
        print('GridSearch 最优参数：', search.best_params_)
        print('GridSearch 最优分数：%0.4lf' % search.best_score_)
        print('准确率: %0.4lf' % accuracy_score(test_y, predict_y))

if __name__ == '__main__':
    pca_classifier(classifiers1, classifiers_names1, classifiers_param_grid1, train_x, train_y, test_x)
    mm_classifier(classifiers2, classifiers_names2, classifiers_param_grid2, train_x, train_y, test_x)