import spacy
from nltk.stem.porter import PorterStemmer
from gensim.utils import simple_preprocess

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

# Define a sample text to tokenize
#text = "This is an example sentence. It contains multiple words."
with open("","r") as f:
  text =f.read()
print(text)

# Tokenize the text using spaCy
doc = nlp(text)

# Print each token and its part of speech
for token in doc:
    print(f"Token: {token.text}  POS: {token.pos_}")
# Remove punctuation tokens from the document
doc_without_punct = [token.text for token in doc if not token.is_punct]

# Join the remaining tokens back into a single string
text_without_punct = " ".join(doc_without_punct)

# Print the resulting text
print(text_without_punct)
doc1 = nlp(text_without_punct)
lemmatized_tokens = [token.lemma_ for token in doc1]

# Join the lemmatized tokens back into a single string
lemmatized_text = " ".join(lemmatized_tokens)
# Print the resulting text
print(lemmatized_text)
doc2 = nlp(lemmatized_text)

# Remove the stop words from the tokenized text
tokens_without_stopwords = [token.text for token in doc2 if not token.is_stop]

# Join the non-stopword tokens back into a single string
text_without_stopwords = " ".join(tokens_without_stopwords)

# Print the resulting text
print(text_without_stopwords)

from nltk.stem.porter import PorterStemmer
from gensim.utils import simple_preprocess

# Define a sample text to stem
#with open("/content/text.txt","r") as f:
  #text =f.read()

# Tokenize the text using Gensim's simple_preprocess function
tokens = simple_preprocess(text_without_stopwords)

# Define a stemmer object
stemmer = PorterStemmer()

# Stem each token in the text
stemmed_tokens = [stemmer.stem(token) for token in tokens]

# Join the stemmed tokens back into a single string
stemmed_text = " ".join(stemmed_tokens)

# Print the resulting text
print(stemmed_text)
