from gensim import corpora, models, similarities
# Define some sample documents
with open("/content/pf-source.txt","r") as f:
  doc1 =f.read()
with open("/content/pf-2.txt","r") as f:
  doc2 =f.read()
#doc2 = "This document is the second document"
with open("/content/pf-3.txt","r") as f:
  doc3=f.read() 
with open("/content/pf-4.txt","r") as f:
  doc4=f.read()
with open("/content/pf-5.txt","r") as f:
  doc5=f.read()
with open("/content/ps-6.txt","r") as f:
  doc6=f.read()
# Create a corpus of documents
documents = [doc1, doc2, doc3, doc4,doc5,doc6]
text_corpus = [doc.split() for doc in documents]
dictionary = corpora.Dictionary(text_corpus)
corpus = [dictionary.doc2bow(text) for text in text_corpus]

# Train a TF-IDF model on the corpus
tfidf_model = models.TfidfModel(corpus)
tfidf_corpus = tfidf_model[corpus]

# Create a similarity index for the TF-IDF corpus
similarity_index = similarities.MatrixSimilarity(tfidf_corpus)

# Define a sample query
with open("/content/pf-source.txt","r") as f:
  query = f.read()

# Convert the query to a bag-of-words vector using the corpus dictionary
query_vec = dictionary.doc2bow(query.lower().split())

# Calculate the similarities between the query vector and each document in the corpus
similarities = similarity_index[tfidf_model[query_vec]]

# Sort the documents by similarity
result_docs = sorted(enumerate(similarities), key=lambda item: -item[1])

# Print the ranked results
for doc_id, sim_score in result_docs:
    print(f"Document {doc_id}:  (Similarity score: {sim_score:.3f})")
