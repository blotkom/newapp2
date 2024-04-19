!pip install flask transformers torch
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

app = Flask(__name__)

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()  # Set the model to evaluation mode

def text_to_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = torch.mean(outputs.last_hidden_state, dim=1).squeeze()
    embeddings = embeddings.numpy()
    embeddings /= np.linalg.norm(embeddings)
    return embeddings.tolist()

@app.route('/', methods=['POST', 'GET'])
def generate_embeddings():
    if request.method == 'POST':
        # Get text input from the POST request
        request_data = request.get_json()
        text = request_data.get('text')

        if text:
            try:
                # Generate BERT embeddings for the provided text
                embeddings = text_to_embeddings(text)
                # Return embeddings as JSON response
                return jsonify({'embeddings': embeddings}), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'Text not provided in the request.'}), 400
    elif request.method == 'GET':
        return "hhh u are s", 200
    else:
        return jsonify({'error': 'Method not allowed.'}), 405

if __name__ == "__main__":
    app.run()
