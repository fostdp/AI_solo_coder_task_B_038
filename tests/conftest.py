import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "microservices"))
sys.path.insert(0, str(ROOT_DIR / "backend"))

os.environ["CONFIG_PATH"] = str(ROOT_DIR / "config")
os.environ["REDIS_HOST"] = "localhost"
os.environ["REDIS_PORT"] = "6379"
