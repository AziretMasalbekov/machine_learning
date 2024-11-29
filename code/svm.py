import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

df_names = pd.read_csv('../data/names.csv', low_memory=False)

vectorizer_forename = TfidfVectorizer(analyzer='char', ngram_range=(1,2))

X_forename = vectorizer_forename.fit_transform(df_names['forename'])

y_combined = df_names['gender']

X_train, X_test, y_train, y_test = train_test_split(X_forename, y_combined, test_size=0.2, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Linear Kernel
steps = [('svc', SVC(kernel='linear'))]
svcL_pipeline = Pipeline(steps)
svcL_pipeline.fit(X_train, y_train)
print("Linear Kernel:\n", classification_report(y_test, svcL_pipeline.predict(X_test)))

# Polynomial Kernel
steps = [('svc', SVC(kernel='poly', degree=3))]
svcPoly_pipeline = Pipeline(steps)
svcPoly_pipeline.fit(X_train, y_train)
print("Polynomial Kernel:\n", classification_report(y_test, svcPoly_pipeline.predict(X_test)))

# RBF Kernel
steps = [('svc', SVC(kernel='rbf', class_weight='balanced'))]
svcRBF_pipeline = Pipeline(steps)
svcRBF_pipeline.fit(X_train, y_train)
print("RBF Kernel:\n", classification_report(y_test, svcRBF_pipeline.predict(X_test)))