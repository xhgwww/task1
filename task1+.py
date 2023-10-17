import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler

# 创建StandardScaler对象
scaler = StandardScaler()

# 创建RandomUnderSampler对象
Under_sampler = RandomUnderSampler(random_state=1)

# 读取数据集
data_train = pd.read_csv("C:\\Users\\hlc\\Desktop\\jotang\\Task 1+\\train_data .csv")
data_test = pd.read_csv("C:\\Users\\hlc\\Desktop\\jotang\\Task 1+\\test_data.csv")

# 提取特征和目标变量
X_train = data_train.drop("label", axis=1)  # 特征变量
y_train = data_train["label"]  # 目标变量
X_test = data_test.drop("label", axis=1)  # 特征变量
y_test = data_test["label"]  # 目标变量

# 创建一个SimpleImputer对象，用众数填充NaN值
imputer = SimpleImputer(strategy='most_frequent')

# 在训练数据和测试数据上分别进行填充
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# 对训练集和测试集进行标准化
X_train_standardized = scaler.fit_transform(X_train)
X_test_standardized = scaler.transform(X_test)

# 在训练集上进行欠采样
X_train_standardized, y_train = Under_sampler.fit_resample(X_train_standardized, y_train)

# 创建决策树分类器对象
clf = DecisionTreeClassifier(max_depth=8, min_samples_split=4, min_samples_leaf=7)

# 拟合（训练）决策树分类器
clf.fit(X_train_standardized, y_train)

# 在测试集上进行预测
y_pred_train = clf.predict(X_train_standardized)
y_pred_test = clf.predict(X_test_standardized)

# 计算和显示准确率
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)
print('训练集准确率:', accuracy_train)
print(classification_report(y_train, y_pred_train))
print('测试集准确率：', accuracy_test)
print(classification_report(y_test, y_pred_test))
