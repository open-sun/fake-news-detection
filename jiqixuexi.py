import pandas as pd
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import nlpaug.augmenter.word as naw
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
def get_stopword_list(file):
    with open(file, 'r', encoding='utf-8') as f:    
        stopword_list = [word.strip('\n') for word in f.readlines()]
    return stopword_list
punc=r'~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{} 😭😇😡🤤😅😊🤦‍♀️😉🤞✌️🤷‍♂️🤷‍♀️🤦‍♂️😘😂🤣❤️😍😒👌💕👍🙌😁🤥🤡🤫🤭🧐🤓'
stopwords = get_stopword_list('stop_words.txt')
def trans(data):
    cutwords = list(jieba.lcut_for_search(data))
    final_cutwords = ''
    for word in cutwords:
        if word not in  punc and word not in stopwords:
            final_cutwords += word + ' '
    return final_cutwords
train_df=pd.read_csv('train1.csv')
test_df=pd.read_csv('newtest2.csv')
print(train_df.columns)
t = pd.DataFrame(train_df.astype(str))
train_df["data"] = t["content"]
t = pd.DataFrame(train_df.astype(str))
train_df["data"] = t["data"].apply(trans)
train_data = train_df["data"]
X_train = train_data
y_train = np.asarray(train_df["label"])
t = pd.DataFrame(test_df.astype(str))
test_df["data"] = t["Ofiicial Account Name"]+t["Title"]
t = pd.DataFrame(test_df.astype(str))
test_df["data"] = t["data"].apply(trans)
print(X_train[0])
X_test = test_df["data"]
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print(X_test_tfidf)

x1train, x1test, y1train, y1test = train_test_split(X_train_tfidf, y_train, test_size=0.2, random_state=42)

param_grid = {'alpha': [0.01,0.1,1]}
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5)
grid_search.fit(x1train, y1train)
best_alpha = grid_search.best_params_['alpha']
print(best_alpha)
nb_classifier = MultinomialNB(alpha=best_alpha)
nb_classifier.fit(x1train, y1train)
y_pred1 = nb_classifier.predict(x1test)
accuracy = accuracy_score(y1test, y_pred1)
print(f"Accuracy: {accuracy:.2f}")
precision = precision_score(y1test, y_pred1)
print(f"Precision: {precision:.2f}")
recall = recall_score(y1test, y_pred1)
print(f"Recall: {recall:.2f}")
f1=f1_score(y1test,y_pred1)
print(f'f1_sscore:{f1:.2f}')
'''
#贝叶斯
param_grid = {'alpha': [0.01,0.1,1]}
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5)
grid_search.fit(X_train_tfidf, y_train)
best_alpha = grid_search.best_params_['alpha']
print(best_alpha)
nb_classifier = MultinomialNB(alpha=best_alpha)
nb_classifier.fit(X_train_tfidf, y_train)
y_pred = nb_classifier.predict(X_test_tfidf)
column=['label']
preds = y_pred
predictions =[]
for i in preds:
    predictions.append(i)
print(len(predictions))
submission = pd.DataFrame({'id': test_df['id'],'label':predictions})
submission.to_csv('submit_Bayestest.csv',index=False)
'''
'''
param_grid = {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0,0.05,0.2,0.3,0.4,0.6,0.7,0.8,0.9,0.07,0.03,1.2,1.4,1.7]} 
grid_search = GridSearchCV(Ridge(), param_grid, cv=5)
grid_search.fit(X_train_tfidf, y_train)
best_alpha = grid_search.best_params_['alpha']
ridge_model = Ridge(alpha=best_alpha)
ridge_model.fit(X_train_tfidf, y_train)
column=['label']
y_pred = ridge_model.predict(X_test_tfidf)
preds = y_pred
predictions =[]
for i in preds:
    predictions.append(i)
print(len(predictions))
submission = pd.DataFrame({'id': test_df['id'],'label':predictions})
submission.to_csv('submit_ridge6.csv',index=False)
'''


'''

# 定义超参数范围
param_grid = {
    'n_estimators': [50, 100, 200],  # 决策树的数量
    'max_depth': [None, 10, 20, 30],  # 决策树的最大深度
    'min_samples_split': [2, 5, 10],  # 内部节点再划分所需最小样本数
    'min_samples_leaf': [1, 2, 4]  # 叶子节点最少样本数
}
# 创建 GridSearchCV 对象
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)

# 执行网格搜索，找到最佳超参数
grid_search.fit(X_train_tfidf, y_train)

# 获取最佳超参数
best_params = grid_search.best_params_

# 使用最佳参数训练模型
best_rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
best_rf_model.fit(X_train_tfidf, y_train)

# 预测测试集
y_pred= best_rf_model.predict(X_test_tfidf)
preds = y_pred
for i in range(len(preds)):
    if preds[i] > 0.5:
        preds[i] =int(1) 
    else:
        preds[i] = int(0)
predictions =[]
for i in preds:
    predictions.append(i)
print(len(predictions))
submission = pd.DataFrame({'id': test_df['id'],'label':predictions})
submission.to_csv('submit_rf4.csv',index=False)
'''