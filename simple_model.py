from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from prepare_data import X_train, X_test, y_train, y_test

logreg = LogisticRegression()
tree = DecisionTreeClassifier()

logreg.fit(X_train, y_train)
tree.fit(X_train, y_train)

pred_logreg = logreg.predict(X_test)
pred_tree = tree.predict(X_test)

print("Accuracy Logistic Regression:", accuracy_score(y_test, pred_logreg))
print("Accuracy Decision Tree:", accuracy_score(y_test, pred_tree))
