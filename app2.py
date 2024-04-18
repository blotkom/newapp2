# serverless_function.py
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import json

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

def handler(event, context):
    # Get text input from the HTTP request
    text = event['data']['text']

    # Generate BERT embeddings for the provided text
    embeddings = text_to_embeddings(text)

    # Return embeddings as JSON response
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps({'embeddings': embeddings})
    }
