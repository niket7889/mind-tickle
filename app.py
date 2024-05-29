import os
import pinecone
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer, AutoModel
import torch
from flask import Flask, request, jsonify
from bs4 import BeautifulSoup  # Ensure BeautifulSoup is imported
import logging

# Initialize Pinecone
pc = Pinecone(api_key="4c4439bc-22f3-4c24-a92e-b0d7800489a8")
data_folder = "/Users/shikha/Downloads/assignment-summer-intern-2024-main/samples"

# Create an index in Pinecone
index_name = "case-studies"
if index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

# Connect to the index
index = pc.Index(index_name)

# Load pre-trained model and tokenizer from Huggingface
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.numpy().flatten()


def clean_text(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text(separator=' ', strip=True)


# Flask app for API
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)


@app.route('/upload', methods=['POST'])
def upload_documents():
    data_folder = request.json.get('data_folder')
    if not data_folder:
        app.logger.error("Data folder not provided")
        return jsonify({"error": "Data folder not provided"}), 400

    file_list = os.listdir(data_folder)
    documents = []

    for file_name in file_list:
        if file_name.endswith(".txt"):
            file_path = os.path.join(data_folder, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read().strip()
                    clean_content = clean_text(content)
                    documents.append((file_name, clean_content))
            except Exception as e:
                app.logger.error(f"Error reading file {file_path}: {e}")
                return jsonify({"error": f"Error reading file {file_path}"}), 500

    try:
        pinecone_data = [(file_name, embed_text(content).tolist()) for file_name, content in documents]
        index.upsert(vectors=pinecone_data)
    except Exception as e:
        app.logger.error(f"Error upserting data to Pinecone: {e}")
        return jsonify({"error": "Error upserting data to Pinecone"}), 500

    return jsonify({"message": f"Upserted {len(pinecone_data)} documents to Pinecone."}), 200


@app.route('/docs', methods=['POST'])
def query_documents():
    query_text = request.json.get('query')
    if not query_text:
        app.logger.error("Query text not provided")
        return jsonify({"error": "Query text not provided"}), 400

    try:
        query_embedding = embed_text(query_text)
    except Exception as e:
        app.logger.error(f"Error embedding query text: {e}")
        return jsonify({"error": "Error embedding query text"}), 500

    try:
        query_response = index.query(
            vector=query_embedding.tolist(),
            top_k=5
        )
    except Exception as e:
        app.logger.error(f"Error querying Pinecone index: {e}")
        return jsonify({"error": "Error querying Pinecone index"}), 500

    results = []
    for match in query_response['matches']:
        matched_id = match['id']
        matched_score = match['score']
        try:
            file_path = os.path.join(data_folder, matched_id)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip().replace('\n', ' ')
                results.append({
                    "file_name": matched_id,
                    "score": matched_score,
                    "content": content
                })
        except Exception as e:
            app.logger.error(f"Error reading matched file {file_path}: {e}")
            return jsonify({"error": f"Error reading matched file {file_path}"}), 500

    return jsonify({"results": results}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

#docker build -t your-image-name .

# docker run -p 5000:5000 --name your-container-name your-image-name
