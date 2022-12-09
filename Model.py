import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split



train = pd.read_csv(r'C:\Users\NADY\Desktop\tit\train.csv')
trainy = train['Survived']
train = train.drop(['Ticket','Cabin','Name','PassengerId','Survived','Fare' ], axis = 1) 

#Convert categorical data to numerical data
lb = LabelEncoder()
train['Sex']      = lb.fit_transform(train['Sex'])
train['Embarked'] = lb.fit_transform(train['Embarked'])
train = train.fillna(train.mean())

Xtrain, Xtest, Ytrain, Ytest = train_test_split( train, trainy, test_size=0.3, random_state=6)


logreg = LogisticRegression()
logreg.fit(Xtrain, Ytrain)
acc_log = round(logreg.score(Xtest, Ytest) * 100, 2)
print('LogisticRegression score is : ',acc_log )


print('-----------')

decision_tree = DecisionTreeClassifier()
decision_tree.fit(Xtrain, Ytrain)
acc_decision_tree = round(decision_tree.score(Xtest, Ytest) * 100, 2)
print( 'DecisionTreeClassifier Score is : ' , acc_decision_tree)

print('-----------')

knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(Xtrain, Ytrain)
acc_knn = round(knn.score(Xtest, Ytest) * 100, 2)
print('KNeighborsClassifier Score is : ', acc_knn)

print('-----------')

gaussian = GaussianNB()
gaussian.fit(Xtrain, Ytrain)
acc_gaussian = round(gaussian.score(Xtest, Ytest) * 100, 2)
print('GaussianNB Score is : ', acc_gaussian)

print('-----------')

svc = SVC()
svc.fit(Xtrain, Ytrain)
acc_svc = round(svc.score(Xtest, Ytest) * 100, 2)
print('SVC Score is : ', acc_svc)

print('-----------')

perceptron = Perceptron()
perceptron.fit(Xtrain, Ytrain)
Y_pred = perceptron.predict(Xtest)
acc_perceptron = round(perceptron.score(Xtest, Ytest) * 100, 2)
print('Perceptron Score is : ', acc_perceptron)

print('-----------')

sgd = SGDClassifier()
sgd.fit(Xtrain, Ytrain)
acc_sgd = round(sgd.score(Xtest, Ytest) * 100, 2)
print('SGDClassifier Score is : ', acc_sgd)

print('-----------')

random_forest = RandomForestClassifier(n_estimators=10)
random_forest.fit(Xtrain, Ytrain)
acc_random_forest = round(random_forest.score(Xtest, Ytest) * 100, 2)
print('RandomForestClassifier Score is : ', acc_random_forest)

