import os

REQUIREMENTS_FILE = "setup_for_collab.txt"
os.system(f"poetry export --without-hashes --format=requirements.txt > {REQUIREMENTS_FILE}")

with open(REQUIREMENTS_FILE, "r") as f:
    file_lines = [f"!pip install {x}" for x in f.readlines()]

with open(REQUIREMENTS_FILE, "w") as f:
    f.writelines(file_lines)
