import warnings
warnings.simplefilter("ignore", category=FutureWarning)

import os
import streamlit as st
from neo4j import GraphDatabase
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import Neo4jVector
from transformers import AutoTokenizer, AutoModel
import torch

# Hugging Face API Setup
API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN")
MISTRAL_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
client = InferenceClient(api_key=API_TOKEN, )

# Driver neo4j 
driver = GraphDatabase.driver(
        os.environ['NEO4J_URI'], 
        auth=(os.environ['NEO4J_USERNAME'], os.environ['NEO4J_PASSWORD'])
    )

# Custom Embedding Class
class CustomHuggingFaceEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_text(self, text):
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        except Exception as e:
            print(f"Error during tokenization: {e}")
            return []
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    
    def embed_query(self, text):
        return self.embed_text(text)
    
    def embed_documents(self, text):
        return self.embed_text(text)

# Function to set up the Neo4j Vector Index
@st.cache_resource 
def setup_vector_index():
    return Neo4jVector.from_existing_graph(
        CustomHuggingFaceEmbeddings(),
        url=os.environ['NEO4J_URI'],
        username=os.environ['NEO4J_USERNAME'],
        password=os.environ['NEO4J_PASSWORD'],
        index_name='articles',
        node_label="Article",
        text_node_properties=['name', 'abstract'],
        embedding_node_property='embedding',
    )

# Query Mistral
def query_from_mistral(context: str, user_input: str):
    messages = [
        {"role": "system", "content": f"Use the following context to answer the query:\n{context}"},
        {"role": "user", "content": user_input},
    ]
    completion = client.chat.completions.create(
        model=MISTRAL_MODEL_NAME,
        messages=messages,
        max_tokens=500,
    )
    return completion.choices[0].message["content"]

# Find keywords
def query_article_keywords(name):
    with driver.session() as session:
        query = """
        MATCH (a:Article)-[:CONTAIN]->(k:Keyword)
        WHERE a.name = $name
        RETURN k
        """
        result = session.run(query, name=name)
        return [record["k"] for record in result]

# extract data from retriever response
def extract_data(documents):
    result = []

    for doc in documents:
        publication_date = doc.metadata.get('date_publication', "N/A")
        page_content = doc.page_content.strip().split("\n")
        
        title = "N/A"
        abstract = "N/A"

        for line in page_content:
            if line.lower().startswith("name:"):
                title = line[len("name:"):].strip()
            elif line.lower().startswith("abstract:"):
                abstract = line[len("abstract:"):].strip()

        keywords = query_article_keywords(title)
        keywords = [dict(node)['text'] for node in keywords]

        doc_data = {
            "Publication Date": publication_date,
            "Title": title,
            "Abstract": abstract,
            "keywords": ','.join(keywords)
        }
        result.append(doc_data)

    return result

# Main Streamlit Application
def main():
    st.set_page_config(page_title="Vector Chat with Mistral", layout="centered")
    
    # App description and features
    st.title("🤖 RAG with Mistral")
    st.markdown("""
        ## Description:
        Chat with **Mistral-7B-Instruct** using context retrieved from a **Neo4j** vector index. This app allows you to ask questions, and the assistant will provide real-time, context-driven answers by querying relevant articles and their keywords from the database.
    """)

    st.image(image="image.jpg", caption="Neo4j")

    st.markdown("""
        ## Key Features:
        - **Real-time context search** from a Neo4j vector index.
        - **Integration with Mistral-7B-Instruct model** for natural language processing.
        - **Keyword extraction** from relevant articles for enhanced context-based responses.

        ## GitHub Repository:
        You can find the source code and more information about this app on GitHub: [GitHub Repository Link](https://github.com/yourusername/your-repository-name)
    """)

    # Initialize the vector index
    vector_index = setup_vector_index()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("You:", "")
        submit = st.form_submit_button("Send")

    if submit and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Fetching response..."):
            try:
                context_results = vector_index.similarity_search(user_input, k=5)

                if not context_results:
                    st.warning("No relevant context found. Please refine your query.")
                    response = "I'm sorry, I couldn't find any relevant information to answer your question."
                else:
                    data_dict = extract_data(context_results)

                    # convert to string
                    context = '\n'.join([ 
                        f"Title: {doc['Title']}\n"
                        f"Abstract: {doc['Abstract']}\n"
                        f"Publication Date: {doc['Publication Date']}\n"
                        f"Keywords: {doc['keywords']}"
                        for doc in data_dict
                    ])

                    response = query_from_mistral(context.strip(), user_input)

                st.session_state.messages.append({"role": "bot", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        elif message["role"] == "bot":
            st.markdown(f"**Bot:** {message['content']}")

if __name__ == "__main__":
    main()
