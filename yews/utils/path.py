from pathlib import Path

def get_files_under_dir(path, pattern):
    return [p for p in Paht(path).glob(pattern) if p.is_file()]
