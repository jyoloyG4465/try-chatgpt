import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ.get("API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini", messages=[{"role": "user", "content": "こんにちは"}]
)

print(response.choices[0].message.content)
