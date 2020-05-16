from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
import pydotplus
import shap
shap.initjs()
import xgboost
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
'''
classifiers_names = ['KNN', 'SVM', '决策树', '随机森林', 'AdaBoost', 'XGBoost', '朴素贝叶斯', '逻辑回归', '神经网络']
results = [0.8030, 0.8068, 0.8033, 0.8051, 0.8056, 0.8073, 0.7788, 0.7973, 0.8068]
df = pd.DataFrame({'分类器': classifiers_names, '正确率': results})
sns.barplot(x='分类器', y="正确率", data=df)
plt.title("各分类器准确率比较")
plt.show()
'''
data = pd.read_csv('alipay_huabei_GS1.csv')
df = data.drop(["ID"], axis=1)
x = df.drop(["default.payment.next.month"], axis=1)
y = data["default.payment.next.month"]
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.30, stratify=y, random_state=3000)
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
ss = StandardScaler()
train_xss = ss.fit_transform(train_x)
test_xss = ss.transform(test_x)

sns.set(color_codes=True)
sns.set_style("white")

# 不同分类的特征小提琴图
'''
for i in ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2',
          'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
          'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
          'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']:
    sns.violinplot(data=df, y=i, x="default.payment.next.month", palette="muted", split=True)
    plt.show()
'''
# jupyter 决策树分类图
dt = DecisionTreeClassifier(min_samples_leaf=4, max_depth=4, max_features=2)
dt.fit(train_xss, train_y)
dt_reg = DecisionTreeRegressor(min_samples_leaf=4, max_depth=4, max_features=2)
dt_reg.fit(train_xss, train_y)
DT_dot_data = export_graphviz(dt_reg,
                               out_file=None,
                               feature_names=['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2',
          'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
          'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
          'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
                              )
DT_graph = pydotplus.graph_from_dot_data(DT_dot_data)
DT_graph.write_pdf('DT.pdf')

#XGBoost特征分析
XGB = xgboost.train({"learning_rate": 0.65, 'gamma': 1, 'max_depth': 1, 'min_child_weight': 0.8, 'colsample_bytree': 1,
                     'subsample': 1}, xgboost.DMatrix(x, label=y), 100)
explainer = shap.TreeExplainer(XGB, feature_perturbation="tree_path_dependent")
shap_values = explainer.shap_values(x)
#某一特征影响
#shap.dependence_plot("PAY_0", shap_values, x)
#全部特征影响
#shap.summary_plot(shap_values, x)
#全部特征正影响
#shap.summary_plot(shap_values, x, plot_type="bar")

