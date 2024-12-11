import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

with open('../model.pkl', 'rb') as f:
    mp = pickle.load(f)

with open('../transformer.pkl', 'rb') as f:
    transformer = pickle.load(f)

single_name = ""
vectorizer_forename = TfidfVectorizer(analyzer='char', ngram_range=(1,2))

single_name_vectorized = transformer.transform([single_name])

pckl = mp.predict(single_name_vectorized)
print(f"Predicted Gender (0 = Male, 1 = Female): {pckl[0]}")

pckl2 = mp.predict_proba(single_name_vectorized)
print(f"Probabilities: {pckl2}")

