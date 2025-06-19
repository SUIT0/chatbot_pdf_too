
import os
from google.generativeai import configure
from google.auth import api_key
try:
    credentials = api_key.Credentials(os.environ["GOOGLE_API_KEY"])
    print("Key validated successfully!")
except Exception as e:
    print(f"Invalid key: {str(e)}")
