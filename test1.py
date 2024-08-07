
import streamlit as st
from pymilvus import connections, Collection
from transformers import AutoTokenizer, AutoModel
import torch
import google.generativeai as palm

# Set your PaLM API key here
PALM_API_KEY = 'AIzaSyC516ARpbqD5vwLrgQFYCeNFJ8gf53Pxx4'
palm.configure(api_key=PALM_API_KEY)

# Initialize the Milvus connection
connections.connect(alias="default", host="127.0.0.1", port="19530")

# Load the collection
collection = Collection(name="resume_collection")

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Function to generate embeddings
def generate_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy().flatten()

# Function to generate a response using PaLM
def generate_palm_response(chunks, query):
    prompt = f"Here are some relevant text chunks related to your query:\n\n{chunks}\n\nBased on the information in these chunks, please answer the following query:\n{query}\n\nAnswer:"
    response = palm.generate_text(
        model='models/text-bison-001',
        prompt=prompt,
        candidate_count=1,  # Number of responses to generate
        temperature=0.7,  # Adjust temperature for response variation
        max_output_tokens=800  # Maximum tokens in the response
    )
    return response.result

# Streamlit app
st.title("Hi! I am Chatbot Representative. Ask me anything about Santosh.")

query_text = st.text_input("Enter your query text:", "")

if st.button("Search and Generate Response"):
    if query_text:
        # Generate embedding for the query text
        query_embedding = generate_embedding(query_text).tolist()

        # Perform a similarity search
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }

        results = collection.search(
            data=[query_embedding],  # List of query embeddings
            anns_field="embedding",  # Field name of the embeddings
            param=search_params,
            limit=10,  # Number of similar entries to retrieve
            expr=None,  # Optional filter expression
            output_fields=["id", "embedding", "text_chunk"]
        )

        # Combine the retrieved text chunks into a single context
        retrieved_text = "\n\n".join([result.text_chunk for result in results[0]])

        # st.write("Retrieved Text Chunks for Context:")
        # st.write(retrieved_text)

        # Generate the response using PaLM
        human_like_answer = generate_palm_response(retrieved_text, query_text)

        # st.write("Human-like Answer:")
        st.write(human_like_answer)
    else:
        st.write("Please enter a query text.")
