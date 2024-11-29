import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df_names = pd.read_csv('../data/names.csv', low_memory=False)

vectorizer_forename = TfidfVectorizer(analyzer='char', ngram_range=(1,2))

X_forename = vectorizer_forename.fit_transform(df_names['forename'])

y_combined = df_names['gender']

X_train, X_test, y_train, y_test = train_test_split(X_forename, y_combined, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=42)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))