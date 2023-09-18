# my-gpt-backend

Python backend for My-GPT.

### Requirements:
- CMake and Make
- Python 3.8+ and Pip 3
- RabbitMQ
- CUDA Toolkit and related drivers (if your system has a GPU)
- [Orca-Mini-3B model file](https://huggingface.co/juanjgit/orca_mini_3B-GGUF/blob/main/orca-mini-3b.q4_0.gguf)

### Setup:
```
git clone https://github.com/aadarsh-ram/my-gpt-backend.git
cd my-gpt-backend

python -m venv venv
. venv/bin/activate

chmod +x ./requirements-setup.sh
./requirements-setup.sh

python3 model_inference.py
```
Note: Download the model file and store it in the root folder.