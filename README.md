# my-gpt-backend

Python backend for My-GPT.

### Requirements:
- CMake and Make
- Python 3.8+ and Pip 3
- RabbitMQ
- CUDA Toolkit and related drivers (if your system has a GPU)
- [Orca-Mini-3B model file](https://huggingface.co/TheBloke/orca_mini_3B-GGML/blob/main/orca-mini-3b.ggmlv3.q5_0.bin)

### Setup:
```
git clone https://github.com/aadarsh-ram/my-gpt-backend.git
cd my-gpt-backend

python -m venv venv
. venv/bin/activate

chmod +x ./requirements-setup.sh
./requirements-setup.sh
```

- Download the model file, convert it to GGUF format using [this script](https://github.com/ggerganov/llama.cpp/blob/master/convert-llama-ggml-to-gguf.py).
- Update .env file with the path to the model file in the `MODEL_PATH` variable. Make any necessary changes for the RabbitMQ server.
- Run the backend with `python3 app.py`