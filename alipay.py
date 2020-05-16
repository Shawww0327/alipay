from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
                AdaBoostClassifier(),
                XGBClassifier()]

# 分类器名称
classifiers_names1 = ['kneighborsClassifier',
                      'svc',
                      'decisionTreeClassifier',
                      'randomForestClassifier',
                      'adaBoostClassifier',
                      'XGBClassifier']

# 分类器参数
classifiers_param_grid1 = [{'kneighborsClassifier__n_neighbors': range(10, 30, 2),
                            'kneighborsClassifier__weights': ['uniform', 'distance'],
                            'kneighborsClassifier__metric': ['euclidean', 'manhattan', 'chebyshev']},
                           {'svc__C': [0.1, 1],
                            'svc__gamma': [1/3.2, 0.5, 1/1.6]},
                           {'decisionTreeClassifier__max_depth': range(2, 10),
                            'decisionTreeClassifier__min_samples_leaf': range(1, 6),
                            'decisionTreeClassifier__max_features': ['log2', 'sqrt', 2]},
                           {'randomForestClassifier__n_estimators': range(51, 101, 10),
                            'randomForestClassifier__max_depth': range(3, 7),
                            'randomForestClassifier__max_features': [1]},
                           {'adaBoostClassifier__n_estimators': range(50, 150, 10),
                            'adaBoostClassifier__learning_rate': [0.3, 0.5, 0.7]},
                           {'XGBClassifier__learning_rate': [0.5, 0.6, 0.7],
                            'XGBClassifier__gamma': [0.5, 1, 1.5],
                            'XGBClassifier__max_depth': [1],
                            'XGBClassifier__min_child_weight': [0.8],
                            'XGBClassifier__subsample': [1],
                            'XGBClassifier__colsample_bytree': [1]}]

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
        accuracy_score_list.append(float('%0.4lf' % accuracy_score(test_y, predict_y)))

# 需要Max_Min的分类器
classifiers2 = [MultinomialNB(),
                LogisticRegression(),
                MLPClassifier()]

# 分类器名称
classifiers_names2 = ['MultinomialNB',
                      'LogisticRegression',
                      'MLPClassifier']

# 分类器参数
classifiers_param_grid2 = [{'MultinomialNB__alpha': [1]},
                           {'LogisticRegression__penalty': ['l2'],
                            'LogisticRegression__max_iter': [1000],
                            'LogisticRegression__C': [1, 10, 100],
                            'LogisticRegression__class_weight': ['None']},
                           #{'MLPClassifier__hidden_layer_sizes': [(100, 50), (100, 60), (100, 70), (100, 80), (100, 90)],
                            #'MLPClassifier__alpha': [0.001, 0.01, 0.1],
                            #'MLPClassifier__learning_rate': ['constant', 'invscaling', 'adaptive']}
                           {'MLPClassifier__hidden_layer_sizes': [(100, 70)],
                            'MLPClassifier__alpha': [0.001],
                            'MLPClassifier__learning_rate': ['constant']}
                           ]

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
        accuracy_score_list.append(float('%0.4lf' % accuracy_score(test_y, predict_y)))

if __name__ == '__main__':
    accuracy_score_list = []
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    pca_classifier(classifiers1, classifiers_names1, classifiers_param_grid1, train_x, train_y, test_x)
    mm_classifier(classifiers2, classifiers_names2, classifiers_param_grid2, train_x, train_y, test_x)
    classifiers_names = ['KNN', 'SVM', '决策树', '随机森林', 'AdaBoost', 'XGBoost', '朴素贝叶斯', '逻辑回归', '神经网络']
    df = pd.DataFrame({'分类器': classifiers_names, '正确率': accuracy_score_list})
    sns.lineplot(x='分类器', y="正确率", data=df)
    plt.title("各分类器准确率比较")
    plt.show()
