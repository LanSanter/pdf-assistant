import openai
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv("API_KEY")

async def openai_answer(conversation):   
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation,
            max_tokens=300,
            temperature=1,
            top_p=0.9
        )
        conversation.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
        message = response['choices'][0]['message']['content']
        return message
    except openai.error.OpenAIError as error:
        print(error)
        return f"Sorry, I encountered an error with status {error.status} - {error.message}. I don't know how to respond."
    
async def embedding_text(text):
    try:
        embedding = openai.Embedding.create(
            model = 'text-embedding-ada-002',
            input = text
        )
        return embedding['data'][0]['embedding']
    except openai.error.OpenAIError as error:
        print(error)
        return f"Sorry, I encountered an error with status {error.status} - {error.message}. I don't know how to respond."
