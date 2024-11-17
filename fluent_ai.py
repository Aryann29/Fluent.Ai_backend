from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import json

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

if __name__ == "__main__":
    user_input = input("Enter your message: ")
    response_text = generate_response(user_input)

    try:
        clean_response = response_text.split("```json")[-1].split("```")[0].strip()
        response_data = json.loads(clean_response)
        print("Response:", response_data["response"])
        print("Corrections:", response_data["corrections"])
        print("Suggestions:")
        for suggestion in response_data["suggestions"]:
            print(f"- {suggestion}")
    except json.JSONDecodeError:
        print("Failed to parse response JSON.")
