import streamlit as st
from transformers import BertTokenizerFast, BertModel, GPT2LMHeadModel, GPT2Tokenizer
import torch

@st.cache_resource
def load_model():
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    return bert_model, bert_tokenizer

# Initialize the GPT-2 model and tokenizer for generation
@st.cache_resource
def load_model_gpt():
    gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
    return gpt_model, gpt_tokenizer

model_bert, tokenizer_bert = load_model()
model_gpt, tokenizer_gpt= load_model_gpt()

def retrieve_documents(query, documents, top_k=3):
    # Tokenize the query
    inputs = tokenizer_bert(query, return_tensors='pt')
    query_embedding = model_bert(**inputs).last_hidden_state.mean(dim=1)
    
    # Retrieve top-k documents based on cosine similarity
    scores = []
    for doc in documents:
        inputs = tokenizer_bert(doc, return_tensors='pt')
        doc_embedding = model_bert(**inputs).last_hidden_state.mean(dim=1)
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
    inputs = tokenizer_gpt(input_text, return_tensors='pt')
    
    # Generate the answer
    outputs = model_gpt.generate(inputs.input_ids, max_length=200, num_return_sequences=1)
    answer = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True)
    
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
