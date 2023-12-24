from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import gc  # Import the garbage collector module

app = Flask(__name__)

# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained("fine_tuned_model")
tokenizer = GPT2Tokenizer.from_pretrained("fine_tuned_model", local_files_only=True)

# Initialize the model and tokenizer on startup
model.eval()
tokenizer.pad_token = tokenizer.eos_token

@app.route('/')
def home():
    return render_template('index.html')

# Endpoint for the chatbot
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('user_input', '')

    if user_input:
        response = generate_response(user_input)
        return jsonify({'response': response})

    return jsonify({'error': 'Invalid input'}), 400

def generate_response(question, max_length=700):
    input_text = "Q: " + question + "\nA:"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate attention mask
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

    # Generate a response
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=5,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

    # Decode the generated response
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Ensure the response is not too long
    if len(response) > max_length:
        response = response[:max_length]
    # Perform garbage collection to free up memory
    gc.collect()
    return response

if __name__ == '__main__':
    app.run(debug=True)
