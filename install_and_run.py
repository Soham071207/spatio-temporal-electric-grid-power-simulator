import os
import time
import subprocess

def install():
    print("Starting install loop...")
    while True:
        try:
            print("Attempting to install PyTorch with CUDA...")
            result = subprocess.run(
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --user", 
                shell=True, 
                capture_output=True, 
                text=True
            )
            print(result.stdout)
            if result.returncode == 0:
                print("PyTorch installed successfully!")
                break
            else:
                print("Install failed. Retrying in 30 seconds...")
                print(result.stderr)
                time.sleep(30)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(30)
    
    print("Installing torch-geometric...")
    subprocess.run("pip install torch-geometric --user", shell=True)

    print("Starting training engine...")
    subprocess.run("python forecasting/train_holiday_focus.py", shell=True)

if __name__ == "__main__":
    install()
