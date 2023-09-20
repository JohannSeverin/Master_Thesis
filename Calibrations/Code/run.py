import os

for file in os.listdir():
    if file == "run.py":
        continue
    if file.endswith(".py"):
        with open(file, "r") as f:
            code = f.read()
            exec(code)
