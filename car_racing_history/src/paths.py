import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def subprocess_environment():
    env = os.environ.copy()
    import_paths = [str(PROJECT_ROOT), env.get("PYTHONPATH")]
    env["PYTHONPATH"] = os.pathsep.join(path for path in import_paths if path)

    return env
