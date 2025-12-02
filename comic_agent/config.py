import os
import json
from dotenv import load_dotenv

# Load environment variables
script_dir = os.path.dirname(os.path.abspath(__file__))
# Point to the .env file in the werewolf directory
env_path = os.path.join(script_dir, ".env")
load_dotenv(dotenv_path=env_path)

class ComicConfig:
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    
    # Load project config
    CONFIG_PATH = os.path.join(script_dir, "project_config.json")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        PROJECT_CONFIG = json.load(f)
    
    OUTPUT_DIR = "output_comics"

