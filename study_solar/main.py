import os
from dotenv import load_dotenv 

print(f'{os.getenv("UPSTAGE_API_KEY")}')