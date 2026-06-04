from pathlib import Path


# Project root
ROOT = Path(__file__).resolve().parent

# Common directories
CONFIG_DIR = ROOT / "config"
ENVIRONMENT_DIR = ROOT / "environment"
PROPAGATORS_DIR = ROOT / "propagators"
TOOLS_DIR = ROOT / "tools"

# Data
DATA_DIR = ROOT / "data"
SCENARIOS_DIR = ROOT / "scenarios"
OUT_DIR = ROOT / "out"

DATA_DIR.mkdir(exist_ok=True)
SCENARIOS_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)


if __name__ == "__main__":
    print("paths.py executed correctly!")