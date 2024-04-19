from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

app = Flask(__name__)

def text_to_embeddings(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = torch.mean(outputs.last_hidden_state, dim=1).squeeze()
    embeddings = embeddings.numpy()
    embeddings /= np.linalg.norm(embeddings)
    return embeddings.tolist()

@app.route('/', methods=['POST'])
def generate_embeddings():
    if request.method == 'POST':
        # Get text input from the POST request
        request_data = request.get_json()
        text = request_data.get('text')

        if text:
            # Generate BERT embeddings for the provided text
            embeddings = text_to_embeddings(text)

            # Return embeddings as JSON response
            return jsonify({'embeddings': embeddings}), 200
        else:
            return jsonify({'error': 'Text not provided in the request.'}), 400
    else:
        return jsonify({'error': 'Method not allowed.'}), 405

if __name__ == "__main__":
    app.run()
