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

steps = [
    ('scaler', StandardScaler(with_mean=False)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]

KNN_pipeline = Pipeline(steps)
KNN_pipeline.fit(X_train, y_train)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

ypred_test = KNN_pipeline.predict(X_test)
mat_clf = confusion_matrix(y_test, ypred_test)
report_clf = classification_report(y_test, ypred_test)

print(mat_clf)
print(report_clf)

ypred_testP = KNN_pipeline.predict_proba(X_test)
auc = roc_auc_score(y_test, ypred_testP[:,1])
print(auc)