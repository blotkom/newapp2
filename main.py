from fastapi import FastAPI, HTTPException
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

app = FastAPI()

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

@app.post("/")
async def generate_embeddings(text: str):
    if text:
        try:
            # Generate BERT embeddings for the provided text
            embeddings = text_to_embeddings(text)
            # Return embeddings as JSON response
            return {'embeddings': embeddings}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail='Text not provided in the request.')
