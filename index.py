import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score

logisticData = pd.read_csv('logistyczna.csv', sep=',')

x = logisticData[['gre', 'gpa', 'rank']]
y = logisticData[['admit']]
x_train, x_test, y_train, y_test = train_test_split(x,   y,   test_size=0.4,  random_state=0)

reg_classifier = LogisticRegression(random_state=0).fit(x, y)

forest = RandomForestClassifier(n_estimators=140, max_leaf_nodes=20, n_jobs=-1, random_state=0)
forestClassifier = forest.fit(x_train, y_train)

reg_train_acc = reg_classifier.score(x_train, y_train)
reg_test_acc = reg_classifier.score(x_test, y_test)

forest_train_score = forest.score(x_train, y_train)
forest_test_score = forest.score(x_test, y_test)


print("Train Reg acc -> ", reg_train_acc)
print("Test Reg acc -> ", reg_test_acc)
print("Train Forest acc -> ", forest_train_score)
print("Test Reg acc -> ", forest_test_score)

reg_pred = reg_classifier.predict(x)
reg_conf_matrix = confusion_matrix(y, reg_pred)

forestPredict = forest.predict(x)
forestCFMatrix = confusion_matrix(y, forestPredict)

reg_precision = precision_score(y, reg_pred)
reg_recall = recall_score(y, reg_pred)
forest_precision = precision_score(y, forestPredict)
forest_recall = recall_score(y, forestPredict)


print("REGRESSION")
print("Precision -> {0:3f}; Recall {1:3f}".format(reg_precision, reg_recall))
print("FOREST")
print("Precision {0:3f}; recall {1:3f}".format(forest_precision, forest_recall))

reg_auc = roc_auc_score(reg_pred, y)
forest_auc = roc_auc_score(forestPredict, y)

print(reg_auc)
print(forest_auc)



