```
██╗     ██╗     ███╗   ███╗███████╗███████╗██████╗ ██╗   ██╗███████╗██╗   ██╗
██║     ██║     ████╗ ████║██╔════╝██╔════╝██╔══██╗██║   ██║██╔════╝██║   ██║
██║     ██║     ██╔████╔██║███████╗█████╗  ██████╔╝██║   ██║█████╗  ██║   ██║
██║     ██║     ██║╚██╔╝██║╚════██║██╔══╝  ██╔══██╗██║   ██║██╔══╝  ╚██╗ ██╔╝
███████╗███████╗██║ ╚═╝ ██║███████║███████╗██║  ██║╚██████╔╝███████╗ ╚████╔╝
╚══════╝╚══════╝╚═╝     ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝  ╚═══╝
```

# LLM Inference Server

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-blue)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![llama.cpp](https://img.shields.io/badge/backend-llama.cpp-green.svg)](https://github.com/ggerganov/llama.cpp)

An LLM inference server optimized for Apple Silicon, started as a fun experiment to see just how much performance I could squeeze out of my MacBook Pro M2.

## What's Inside?

- **Apple Silicon Optimized**: Uses Metal Performance Shaders (MPS) to get the most out of the M2 GPU.
- **Multi-Model Support**: Comes ready to run with `Phi-3`, `Gemma`, and `TinyLlama`.
- **Dynamic Batching**: Groups incoming requests on the fly to maximize throughput, making the server fast and responsive.
- **Real-time Streaming**: Supports live, streaming responses for building interactive applications.
- **Memory-Conscious**: Built to run smoothly on an 8GB unified memory system by using GGUF quantized models.
- **Modern FastAPI Backend**: A fast, modern web framework that provides interactive API documentation right out of the box.

## Hardware Requirements

- Apple M2 chip (optimized for M2, may work on other Apple Silicon)
- 8GB+ unified memory
- macOS Sonoma or later

## Installation

1.  Clone the repository:

    ```bash
    git clone <repository-url>
    cd llm-inference-server
    ```

2.  Install dependencies using Poetry:

    ```bash
    poetry install
    ```

3.  Download the models:

    The script is configured to download all supported models by default.

    ```bash
    chmod +x scripts/download_model.sh
    ./scripts/download_model.sh
    ```

    To download a specific model, pass it as an argument (e.g., `phi3`, `gemma`, `tinyllama`):

    ```bash
    ./scripts/download_model.sh phi3 tinyllama
    ```

    > **Note on Gemma**: The Gemma model is gated and requires a Hugging Face account. For the most reliable download, please log in via the CLI first:
    >
    > ```bash
    > pip install -U "huggingface_hub[cli]"
    > huggingface-cli login
    > ```
    >
    > You must also accept the license on the [Gemma model page](https://huggingface.co/google/gemma-2b-it).

## Configuration

The server uses environment variables for configuration. Create a `.env` file in the root directory and choose which model to run.

```env
# --- Choose a model ---
# MODEL_PATH=./models/phi-3-mini-4k-instruct-q4.gguf
MODEL_PATH=./models/gemma-2b-it.Q4_K_M.gguf
# MODEL_PATH=./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# --- Server and Llama.cpp configuration ---
SERVER_HOST=0.0.0.0
SERVER_PORT=8081
N_GPU_LAYERS=32
N_CTX=4096
N_BATCH=512
MAX_BATCH_SIZE=8
BATCH_TIMEOUT=0.1
```

## Usage

### Start the server:

```bash
poetry run dev
```

### API Documentation

Once running, visit:

- API Documentation: http://localhost:8081/docs
- Alternative docs: http://localhost:8081/redoc

### Example API Call

```bash
curl -X POST "http://localhost:8081/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the capital of France?",
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": false
  }'
```

### Testing the Streaming Feature

The server supports real-time streaming of responses. Use the `-N` flag with `curl` to disable buffering and see the tokens as they are generated.

```bash
curl -N -X POST "http://localhost:8081/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a short story about a robot who discovers music.",
    "max_tokens": 250,
    "stream": true
  }'
```

### Testing the Dynamic Batching Feature

You can test the server's ability to handle concurrent requests and batch them together by sending multiple requests simultaneously. The server will group these into a single batch for efficient processing.

```bash
curl -X POST "http://localhost:8081/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the capital of France?",
    "max_tokens": 50,
    "stream": false
  }' &

curl -X POST "http://localhost:8081/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a short poem about a robot.",
    "max_tokens": 150,
    "stream": false
  }' &

curl -X POST "http://localhost:8081/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain the theory of relativity in simple terms.",
    "max_tokens": 200,
    "stream": false
  }' &

wait
```

## Interactive Chat CLI

This project includes a simple, interactive command-line interface (CLI) to chat with your model directly from the terminal.

### 1. Install CLI Dependencies

Install the extra packages required for the CLI using Poetry:

```bash
poetry install --extras "cli"
```

### 2. Run the Chat

Make sure the inference server is running in a separate terminal. Then, start the chat client:

```bash
poetry run chat
```

You can now chat with the model interactively. Type `exit` or `quit` to end the session.

## Performance Tuning

### Model Selection

- Use Q4_0 or Q4_1 quantization for best memory/quality balance
- Smaller models (3B-7B parameters) work best on 8GB systems
- Consider Phi-3-mini for excellent performance/size ratio

### Batching Configuration

- `MAX_BATCH_SIZE`: Increase for higher throughput, decrease for lower latency
- `BATCH_TIMEOUT`: Lower values reduce latency, higher values improve throughput
- `N_BATCH`: Should match your typical batch size

### Memory Management

- `N_GPU_LAYERS`: Set to -1 to offload all layers to GPU
- `N_CTX`: Reduce if running out of memory
- Monitor memory usage with Activity Monitor

## Development

### Running in Development Mode

The `dev` command in `pyproject.toml` is already configured to run the server in reload mode.

```bash
poetry run dev
```

### Project Structure

```
llm-inference-server/
├── .env
├── LICENSE
├── README.md
├── models/
│   └── (Downloaded models go here)
├── poetry.lock
├── pyproject.toml
├── requirements.txt
├── scripts/
│   └── download_model.sh
└── src/
    ├── __init__.py
    ├── app/
    │   ├── __init__.py
    │   ├── api/
    │   │   └── v1/
    │   │       ├── __init__.py
    │   │       └── routes.py
    │   ├── main.py
    │   ├── schemas/
    │   │   ├── __init__.py
    │   │   ├── request.py
    │   │   └── response.py
    │   └── services/
    │       ├── __init__.py
    │       └── inference_service.py
    ├── cli/
    │   ├── __init__.py
    │   └── main.py
    ├── inference/
    │   ├── __init__.py
    │   ├── engine.py
    │   └── prompt_templates.py
    └── lib/
        └── core/
            ├── __init__.py
            ├── banner.py
            └── config.py
```

## License

MIT
