from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import os

# Initialize Flask app
app = Flask(__name__)

# Load fine-tuned T5 model and tokenizer
t5_model_path = "./t5_paraphraser"
if not os.path.exists(t5_model_path):
    raise FileNotFoundError(f"Fine-tuned model not found at {t5_model_path}")

tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
model = T5ForConditionalGeneration.from_pretrained(t5_model_path)

# Load Sentence-BERT model for semantic similarity
sbert_model = SentenceTransformer("all-MiniLM-L6-V2")

# Function to paraphrase text
def paraphrase(input_text):
    inputs = tokenizer("paraphrase: " + input_text, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=200, num_beams=8,length_penalty = 2.0, early_stopping=True)
    paraphrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return paraphrased_text

# Function to validate semantic similarity
def validate_semantic_similarity(original, paraphrased):
    embeddings = sbert_model.encode([original, paraphrased], convert_to_tensor=True)
    similarity_score = util.cos_sim(embeddings[0], embeddings[1])
    return similarity_score.item()

# Flask endpoint for paraphrasing and similarity validation
@app.route("/paraphrase", methods=["POST"])
def paraphrase_endpoint():
    try:
        # Parse input JSON
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Invalid input. Provide a JSON object with a 'text' field."}), 400
        
        input_text = data["text"]

        # Perform paraphrasing
        paraphrased_text = paraphrase(input_text)

        # Calculate similarity score
        similarity_score = validate_semantic_similarity(input_text, paraphrased_text)

        # Return the response
        return jsonify({
            "original_text": input_text,
            "paraphrased_text": paraphrased_text,
            "similarity_score": similarity_score
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Main entry point for running the app
if __name__ == "__main__":
    app.run(host="192.168.1.48", port=8000, debug=True)

