import os
import subprocess
import sys
import venv

#Name of the virtual environment
env_name = "ray_env"

#Create the virtual environment
if not os.path.exists(env_name):
    print(f"Creating virtual env: {env_name}")
    venv.create(env_name, with_pip=True)
else:
    print(f"Virtual env '{env_name}' already exists.")

#Define paths to pip/python inside the virtual environment
if sys.platform == "darwin":  # macOS
    pip_path = os.path.join(env_name, "bin", "pip")
    python_path = os.path.join(env_name, "bin", "python")
else:  # Linux / HPC
    pip_path = os.path.join(env_name, "bin", "pip")
    python_path = os.path.join(env_name, "bin", "python")

#Upgrade pip and install requirements
print("Upgrading pip, setuptools, and wheel...")
subprocess.run([pip_path, "install", "--upgrade", "pip", "setuptools", "wheel"], check=True)

#Install from requirements.txt
if os.path.exists("requirements.txt"):
    print("Installing dependencies from requirements.txt...")
    subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
    print("Installation complete.")
else:
    print("Error: requirements.txt not found in the current directory.")
    sys.exit(1)

#Print activation hint
print(f"\nDone! Activate the environment with:\n  source {env_name}/bin/activate\n")
