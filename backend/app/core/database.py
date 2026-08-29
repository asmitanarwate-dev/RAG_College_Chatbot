import os
from pymongo import MongoClient
from dotenv import load_dotenv
import certifi

# Load .env from the same folder as this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH)

MONGODB_URL = os.getenv("MONGODB_URL")
DATABASE_NAME = os.getenv("DATABASE_NAME")

if not MONGODB_URL:
    raise ValueError("MONGODB_URL is not set in .env")

if not DATABASE_NAME:
    raise ValueError("DATABASE_NAME is not set in .env")

client = MongoClient(
    MONGODB_URL,
    tls=True,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=15000
)

# Test connection
client.admin.command("ping")

db = client[DATABASE_NAME]

print("MongoDB connected successfully")
print(f"Database: {DATABASE_NAME}")