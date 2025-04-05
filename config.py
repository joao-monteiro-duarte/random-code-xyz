import os
from dotenv import load_dotenv

# Load and set env vars first
load_dotenv()
os.environ["REDIS_HOST"] = os.getenv("REDIS_HOST", "localhost")
os.environ["REDIS_PORT"] = os.getenv("REDIS_PORT", "6380")
os.environ["REDIS_DB"] = "0"
os.environ["REDIS_PASSWORD"] = ""  # Empty since no auth

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "CG-9QJjJ8s1jHGcVBU3qXB9imY3")
YOUTUBE_API_KEYS = os.getenv("YOUTUBE_API_KEYS", "AIzaSyD8LfVIXxQ9IDCKKTAxruyRTWOowcGRikE,AIzaSyALny7-Swi_ChgdUz3ejj9uUi_vumlSmQ4,AIzaSyCgcKppv_-vvVMtvRTXYmldM_IWNMXjGy4,AIzaSyCgDsA0Elo3T__cTfFVu-nSljFNf0Rvzm8,AIzaSyAeIVcX92DPKYEfnpAXI-5uFKMV01KzUhE,AIzaSyDJwZTTAeEhSDDtBD20M8t9vDHY0KfUjLE, AIzaSyBQBfE8ZJtAmla_2Om7knUKOFarBTCF20M").split(",")
VPH_THRESHOLD = 1500
VPH_VIRAL_THRESHOLD = 10000.0
CACHE_FILE = "video_cache.json"
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6380))