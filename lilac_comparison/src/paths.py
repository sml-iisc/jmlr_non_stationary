import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "configs"


def subprocess_environment():
    """Keep this experiment importable in child Python processes."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(PROJECT_ROOT), env.get("PYTHONPATH")) if path
    )

    return env
