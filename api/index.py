from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import json

app = Flask(__name__)
CORS(app)
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # or gemini-1.5-pro
    verbose=True,
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def generate_response(user_input: str):
    prompt = f"""
    You are FluentAI, a friendly English tutor who helps users improve their English through conversation
    Respond to: "{user_input}"
    
    Have a natural conversation while:
    1. Responding in a friendly way dont correct anything in Responding.
    2. Correcting any grammar mistakes
    3. Suggesting alternative words or expressions
    
    Format your response as valid JSON with these keys:
    {{
        "response": "your conversational response",
        "corrections": "any grammar corrections",
        "suggestions": [
            "Instead of 'How are you?', you could say 'How's it going?'",
            "'What's up?'",
            "'How are you feeling?'"
        ]
    }}
    """
    response = llm.invoke(prompt)
    return response.content

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400

        user_input = data['message']
        response_text = generate_response(user_input)

        try:
            clean_response = response_text.split("```json")[-1].split("```")[0].strip()
            response_data = json.loads(clean_response)
        except json.JSONDecodeError:
            return jsonify({'status': 'error', 'message': 'Failed to parse response JSON.'}), 500

        return jsonify({'status': 'success', 'data': response_data})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'message': 'FluentAI is running'
    })

if __name__ == '__main__':
    app.run()
