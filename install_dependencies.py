def install_dependencies():
    import os
    os.system('pip install --quiet bitsandbytes')
    os.system('pip install --quiet git+https://github.com/huggingface/transformers.git')
    os.system('pip install --quiet accelerate')
    os.system('pip install transformers')
    os.system('pip install torch')
    os.system('pip install datasets')
    os.system('pip install accelerate -U')
    os.system('pip install transformers[torch] accelerate -U')
    os.system('pip install bitsandbytes')