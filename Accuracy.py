relevant_docs = [doc1,doc2,doc3,doc4,doc5,doc6] # the relevant documents are assumed to be doc1, doc2, and doc7
num_relevant_docs = len(relevant_docs)
num_correct = 0
for doc_id, sim_score in result_docs:
    if documents[doc_id] in relevant_docs:
        num_correct += 1
accuracy = num_correct / num_relevant_docs
print(f"Accuracy: {accuracy:.2f}")
