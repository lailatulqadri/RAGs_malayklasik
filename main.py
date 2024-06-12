import streamlit as st
from transformers import BertTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
import torch

@st.cache_resource
def load_model():
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    return model, tokenizer

# Initialize the GPT-2 model and tokenizer for generation
@st.cache_resource
def load_model_gpt():
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    return model, tokenizer

def retrieve_documents(query, documents, top_k=3):
    # Tokenize the query
    inputs = bert_tokenizer(query, return_tensors='pt')
    query_embedding = bert_model(**inputs).last_hidden_state.mean(dim=1)
    
    # Retrieve top-k documents based on cosine similarity
    scores = []
    for doc in documents:
        inputs = bert_tokenizer(doc, return_tensors='pt')
        doc_embedding = bert_model(**inputs).last_hidden_state.mean(dim=1)
        score = torch.nn.functional.cosine_similarity(query_embedding, doc_embedding)
        scores.append(score.item())
    
    top_docs_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    top_docs = [documents[i] for i in top_docs_indices]
    
    return top_docs

def generate_answer(query, documents):
    # Retrieve relevant documents
    retrieved_docs = retrieve_documents(query, documents)
    context = " ".join(retrieved_docs)
    
    # Combine the query and context for generation
    input_text = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    inputs = gpt2_tokenizer(input_text, return_tensors='pt')
    
    # Generate the answer
    outputs = gpt2_model.generate(inputs.input_ids, max_length=200, num_return_sequences=1)
    answer = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer

# Sample documents
documents = [
    "The capital of France is Paris. It is known for the Eiffel Tower.",
    "The largest planet in our solar system is Jupiter. It is a gas giant.",
    "The Great Wall of China is one of the seven wonders of the world.",
]

# Streamlit interface
st.title("Retrieval-Augmented Generation (RAG) Example")

query = st.text_input("Enter your question:")
if st.button("Generate Answer"):
    if query:
        answer = generate_answer(query, documents)
        st.write("Answer:", answer)
    else:
        st.write("Please enter a question.")
