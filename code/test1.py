import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

with open('../model.pkl', 'rb') as f:
    mp = pickle.load(f)

with open('../transformer.pkl', 'rb') as f:
    transformer = pickle.load(f)

single_name = "Jack"
vectorizer_forename = TfidfVectorizer(analyzer='char', ngram_range=(1,2))

single_name_vectorized = transformer.transform([single_name])

pckl = mp.predict(single_name_vectorized)
if pckl[0] == 1:
    print("Predicted Gender: Female")
if pckl[0] == 0:
    print("Predicted Gender: Male")

