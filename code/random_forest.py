import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

df_names = pd.read_csv('../data/names.csv')

vectorizer_forename = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))
X_forename = vectorizer_forename.fit_transform(df_names['forename'])
y_combined = df_names['gender']


X_train, X_test, y_train, y_test = train_test_split(X_forename, y_combined, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


param_dist = {
    'n_estimators': [100, 50, 150],
    'max_depth': [10, 15, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [2, 4],
    'max_features': ['sqrt', 'log2', None],
}

pipeline = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('rf', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=20,
    scoring='roc_auc',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("\nBest Parameters:", random_search.best_params_)
best_rf = random_search.best_estimator_

ypred_test = best_rf.predict(X_test)
print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, ypred_test))

print("\nTest Classification Report:")
print(classification_report(y_test, ypred_test))

ypred_testP = best_rf.predict_proba(X_test)
auc_test = roc_auc_score(y_test, ypred_testP[:, 1])
print(f"Test AUC: {auc_test:.2f}")

ypred_train = best_rf.predict(X_train)
print("\nTrain Confusion Matrix:")
print(confusion_matrix(y_train, ypred_train))

print("\nTrain Classification Report:")
print(classification_report(y_train, ypred_train))

ypred_trainP = best_rf.predict_proba(X_train)
auc_train = roc_auc_score(y_train, ypred_trainP[:, 1])
print(f"Train AUC: {auc_train:.2f}")
