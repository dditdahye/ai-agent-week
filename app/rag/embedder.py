from openai import OpenAI
from dotenv import load_dotenv
from app.config import EMBED_MODEL

load_dotenv()

client = OpenAI()
MODEL = EMBED_MODEL

def get_embedding(text: str) -> list[float]:
    resp = client.embeddings.create(
        model=MODEL,
        input=text
    )
    return resp.data[0].embedding