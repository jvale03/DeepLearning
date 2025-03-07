import subprocess

scripts = [
    "merge_datasets.py",
    "split_datasets.py",
    "data_sample.py",
    "process_text.py"
]


def executarScripts():
    for script in scripts:
        print(f"A executar {script}...")
        result = subprocess.run(["python3", script], capture_output=True, text=True)
        
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"Erro ao executar {script}:\n{result.stderr}")
            break


if __name__ == "__main__":
    executarScripts()