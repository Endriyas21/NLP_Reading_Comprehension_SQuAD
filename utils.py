import subprocess

def check_gpu():
    try:
        gpu_info = subprocess.check_output(['nvidia-smi'], encoding='utf-8')
        print(gpu_info)
    except subprocess.CalledProcessError as e:
        print('Not connected to a GPU')