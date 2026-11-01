"""Central config. Imported FIRST everywhere: sets env flags before crewai/chromadb load."""
import os
from pathlib import Path

# --- zero-network / zero-key safety flags (Task: telemetry + MOCK_LLM) ---
for k, v in {"CREWAI_DISABLE_TELEMETRY": "true", "OTEL_SDK_DISABLED": "true",
             "CREWAI_DISABLE_TRACKING": "true", "CREWAI_TRACING_ENABLED": "false",
             "ANONYMIZED_TELEMETRY": "False", "MOCK_LLM": "true"}.items():
    os.environ.setdefault(k, v)

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"; ART.mkdir(exist_ok=True)
LOG_DIR = Path(os.environ.get("NYKAA_LOG_DIR", ROOT / "logs")); LOG_DIR.mkdir(exist_ok=True)
CHROMA_DIR = Path(os.environ.get("NYKAA_CHROMA_DIR", ROOT / "chroma_db"))
CALIB_FILE = Path(os.environ.get("NYKAA_CALIB_FILE", ART / "calibration.json"))

TOP_K = 3
MAX_QUERY_CHARS = 2000          # hard input-size guard
TOKEN_BUDGET = 3500             # per-request token cap (Runtime governance layer)
API_KEY = os.environ.get("API_KEY", "")   # if set, /add-document requires X-API-Key
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "60"))
