from __future__ import annotations
import re
import os

def sorted_alphanumeric(data: list[str]) -> list[str]:
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def get_sorted_alpanumeric_files(data_dir: list[str], extensions: list[str] | tuple[str] | set[str]) -> list[str]:
    ret_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            ext = file.split('.')[-1]
            if ext in extensions:
                ret_files.append(os.path.join(root, file))
    return sorted_alphanumeric(ret_files)
