import faiss
import numpy as np
import ollama

# Initialize Ollama for embeddings and response generation
ollama_client = ollama.Client()  # Ensure Ollama is running locally

# Example documents to index
documents = [
    "This is the first document. It explains how module1.py works.",
    "The second document describes module2.py with compatibility information.",
    "This document talks about performance optimization of the framework."
]

# Generate embeddings for your documents using Ollama
def generate_embeddings(texts):
    embeddings = []
    for text in texts:
        response = ollama.embed(
            model="mxbai-embed-large",
            input=text,
        )
        embedding = response["embeddings"]  # Correct key is "embedding"
        embeddings.append(embedding)
    return np.vstack(embeddings)  # Stack into a 2D array

# Create FAISS index
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # Assuming embeddings have consistent dimensions
    index = faiss.IndexFlatL2(dimension)  # Use L2 distance metric
    index.add(embeddings)  # Add embeddings to index
    return index

# Retrieve relevant chunks using FAISS
def retrieve_relevant_chunks(query, index, documents, num_results=2):
    response = ollama.embed(model="mxbai-embed-large", input=query)
    query_embedding = np.array(response["embeddings"]).reshape(1, -1)  # Reshape to 2D
    distances, indices = index.search(query_embedding, num_results)
    return [documents[i] for i in indices[0]]

# Generate a context-aware response using Ollama
def generate_response(query, relevant_chunks):
    context = " ".join(relevant_chunks)
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = ollama.generate(model="llama3.2:3b", prompt=prompt)  # Specify the model
    return response["response"]  # Extract the generated text

# Main workflow
if __name__ == "__main__":
    # Step 1: Embed documents
    embeddings = generate_embeddings(documents)

    # Step 2: Create and store FAISS index
    faiss_index = create_faiss_index(embeddings)

    # Step 3: Handle a user query
    user_query = "How do I optimize module2.py?"
    relevant_chunks = retrieve_relevant_chunks(user_query, faiss_index, documents)

    # Step 4: Generate a response
    final_response = generate_response(user_query, relevant_chunks)
    print(f"Response:\n{final_response}")