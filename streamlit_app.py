# Import libraries
import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer

st.title('❄️ snowflake-arctic-embed')

# Select an embed model
model_list = ['snowflake-arctic-embed-xs', 'snowflake-arctic-embed-s', 'snowflake-arctic-embed-m', 'snowflake-arctic-embed-m-long', 'snowflake-arctic-embed-l']
selected_model = st.selectbox('Select an embed model', model_list)

# Load embed model
@st.cache_resource
def load_tokenizer(input_tokenizer):
    return AutoTokenizer.from_pretrained(f'Snowflake/{input_tokenizer}')

@st.cache_resource
def load_model(input_model):
    return AutoModel.from_pretrained(f'Snowflake/{input_model}', add_pooling_layer=False, trust_remote_code=True, safe_serialization=True)

tokenizer = load_tokenizer(selected_model)
model = load_model(selected_model)
model.eval()

# Query
query_prefix = 'Represent this sentence for searching relevant passages: '
queries  = ['What is Snowflake?', 'Where can I get the best tacos?']
queries_with_prefix = ["{}{}".format(query_prefix, i) for i in queries]
query_tokens = tokenizer(queries_with_prefix, padding=True, truncation=True, return_tensors='pt', max_length=512)

# Document
documents = ['The Data Cloud!', 'Mexico City of Course!']
document_tokens =  tokenizer(documents, padding=True, truncation=True, return_tensors='pt', max_length=512)

# Compute token embeddings
with torch.no_grad():
    query_embeddings = model(**query_tokens)[0][:, 0]
    document_embeddings = model(**document_tokens)[0][:, 0]

# Normalize embeddings
query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
document_embeddings = torch.nn.functional.normalize(document_embeddings, p=2, dim=1)
scores = torch.mm(query_embeddings, document_embeddings.transpose(0, 1))

#Output passages & scores
st.subheader('Output')

for query, query_scores in zip(queries, scores):
    doc_score_pairs = list(zip(documents, query_scores))
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    
    st.write("Query:", query)
    for document, score in doc_score_pairs:
        st.write(score, document)
