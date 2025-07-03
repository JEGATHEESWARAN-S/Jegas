from flask import Flask, request, jsonify
from app.llama_chat import LlamaChatbot

app = Flask(__name__)
chatbot = LlamaChatbot()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"error": "Message is required"}), 400
    
    response = chatbot.generate_response(user_message)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
