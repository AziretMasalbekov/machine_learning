import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

df_names = pd.read_csv('../data/names.csv', low_memory=False)

vectorizer_forename = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))

X_forename = vectorizer_forename.fit_transform(df_names['forename'])

y_combined = df_names['gender']

X_train, X_test, y_train, y_test = train_test_split(X_forename, y_combined, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

param_grid = {
    'knn__n_neighbors': [3, 5, 7, 10],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan', 'minkowski']
}
steps = [
    ('scaler', StandardScaler(with_mean=False)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]

from sklearn.model_selection import GridSearchCV

KNN_pipeline = Pipeline(steps)

grid_search = GridSearchCV(estimator=KNN_pipeline, param_grid=param_grid,
                           scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

grid_search.fit(X_train, y_train)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

best_pipeline = grid_search.best_estimator_

ypred_test = best_pipeline.predict(X_test)
mat_clf = confusion_matrix(y_test, ypred_test)
report_clf = classification_report(y_test, ypred_test)

print("Confusion Matrix (Test):")
print(mat_clf)
print("Classification Report (Test):")
print(report_clf)

ypred_testP = best_pipeline.predict_proba(X_test)
auc = roc_auc_score(y_test, ypred_testP[:, 1])
print("AUC Score (Test):", auc)

ypred_train = best_pipeline.predict(X_train)
mat_clf = confusion_matrix(y_train, ypred_train)
report_clf = classification_report(y_train, ypred_train)

print("Confusion Matrix (Train):")
print(mat_clf)
print("Classification Report (Train):")
print(report_clf)

ypred_trainP = best_pipeline.predict_proba(X_train)
auc = roc_auc_score(y_train, ypred_trainP[:, 1])
print("AUC Score (Train):", auc)