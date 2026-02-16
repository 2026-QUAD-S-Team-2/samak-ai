from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

for m in client.models.list():
    # m.name 예: "models/gemini-..."
    # m.supported_actions 같은 필드가 있으면 같이 출력
    print(m.name)