from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import torch

app = Flask(__name__)

# Load the model and tokenizer
base_model_path = "./lora_base_model"
adapter_path = "./lora_adapter"
base_model = AutoModelForSequenceClassification.from_pretrained(base_model_path)
model = PeftModel.from_pretrained(base_model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
    return jsonify({'prediction': prediction})

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the text from the form
        text = request.form['text']
        if not text.strip():
            return render_template('index.html', error="Please enter some text.")
        
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
        
        # Map prediction to label (assuming 1 = Toxic, 0 = Not Toxic)
        label = "Toxic" if prediction == 1 else "Not Toxic"
        
        return render_template('index.html', prediction=label, text=text)
    
    # If GET request, just show the empty form
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)