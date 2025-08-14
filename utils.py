from typing import List
from pathlib import Path

def get_config_paths() -> List[str]:

    return [str(path) for path in Path("inputs").glob("*.yaml")]