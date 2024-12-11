import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

df_names = pd.read_csv('../data/names.csv')

vectorizer_forename = TfidfVectorizer(analyzer='char', ngram_range=(1,2))

X_forename = vectorizer_forename.fit_transform(df_names['forename'])

y_combined = df_names['gender']

X_train, X_test, y_train, y_test = train_test_split(X_forename, y_combined, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('logReg', LogisticRegression(solver='liblinear'))
])

from sklearn.model_selection import GridSearchCV

param_grid = {
    'logReg__penalty': ['l1', 'l2'],
    'logReg__C': [0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train, y_train)

best_pipeline = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

ypred_test = best_pipeline.predict(X_test)
mat_clf_test = confusion_matrix(y_test, ypred_test)
report_clf_test = classification_report(y_test, ypred_test)

print("Confusion Matrix on Test Data:")
print(mat_clf_test)
print("\nClassification Report on Test Data:")
print(report_clf_test)

ypred_testP = best_pipeline.predict_proba(X_test)
auc_test = roc_auc_score(y_test, ypred_test  [:, 1])
print(f"Test AUC Score: {auc_test}")

single_name = "Myo"

single_name_vectorized = vectorizer_forename.transform([single_name])

predicted_gender = best_pipeline.predict(single_name_vectorized)
print(f"Predicted Gender (0 = Male, 1 = Female): {predicted_gender[0]}")

predicted_probabilities = best_pipeline.predict_proba(single_name_vectorized)
print(f"Probabilities: {predicted_probabilities}")

import pickle

with open('../model.pkl', 'wb') as f:
    pickle.dump(best_pipeline,f)

with open('../transformer.pkl', 'wb') as f:
        pickle.dump(vectorizer_forename, f)



with open('../model.pkl', 'rb') as f:
    mp = pickle.load(f)

pckl = mp.predict(single_name_vectorized)
print(f"Predicted Gender (0 = Male, 1 = Female): {pckl[0]}")
pckl2 = mp.predict_proba(single_name_vectorized)
print(f"Probabilities: {pckl2}")
