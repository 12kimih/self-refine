import re
from pathlib import Path


def get_file_number(dir: Path, name: str) -> int:
    max: int = 0
    pattern = re.compile(name + r"-(\d+)")
    for p in dir.iterdir():
        m = pattern.match(p.stem)
        if m:
            n = int(m.group(1))
            if n > max:
                max = n
    return max + 1
