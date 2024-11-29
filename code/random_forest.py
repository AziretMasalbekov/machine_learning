import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df_names = pd.read_csv('../data/names.csv')

vectorizer_forename = TfidfVectorizer(analyzer='char', ngram_range=(3, 5))

X_forename = vectorizer_forename.fit_transform(df_names['forename'])

y_combined = df_names['gender']

X_train, X_test, y_train, y_test = train_test_split(X_forename, y_combined, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=4, class_weight='balanced', max_depth=10,
                            random_state=42)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')
