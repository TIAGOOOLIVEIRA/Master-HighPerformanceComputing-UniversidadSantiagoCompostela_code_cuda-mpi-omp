import os
import subprocess

#Create env
env_name = "mypython_env"
if not os.path.exists(env_name):
    print(f"Creating virtual environment: {env_name}")
    subprocess.run(["python", "-m", "venv", env_name])
else:
    print(f"Virtual environment '{env_name}' already exists.")

#Activate virtual environment and install dependencies
pip_install_cmd = f"./{env_name}/bin/pip install --upgrade pip && ./{env_name}/bin/pip install -r requirements.txt"
print("Installing dependencies...")
subprocess.run(pip_install_cmd, shell=True, check=True)

print("Environment setup complete! Activate it using:")
print(f"   source {env_name}/bin/activate")
