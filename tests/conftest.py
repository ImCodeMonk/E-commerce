import os, sys, tempfile
from pathlib import Path
_t = Path(tempfile.mkdtemp(prefix="nykaa_test_"))
os.environ.update(NYKAA_CHROMA_DIR=str(_t / "chroma"), NYKAA_LOG_DIR=str(_t), NYKAA_CALIB_FILE=str(_t / "calib.json"),
                  RATE_LIMIT_PER_MIN="100000")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import warnings; warnings.filterwarnings("ignore")
