
# install

1. install uv package manager or another
2. 

    uv sync; source .venv/bin/activate; uv pip install -e .

# usage

## server

    usage: server.py [-h] --model {gemma3-1b,dummy} [--host STR] [--port INT]

    ╭─ options ─────────────────────────────────────────╮
    │ -h, --help        show this help message and exit │
    │ --model {gemma3-1b,dummy}                         │
    │                   the llm to serve (required)     │
    │ --host STR        (default: 0.0.0.0)              │
    │ --port INT        (default: 8080)                 │
    ╰───────────────────────────────────────────────────╯

## client

    usage: client.py [-h] [--host STR] [--port INT]

    ╭─ options ─────────────────────────────────────────╮
    │ -h, --help        show this help message and exit │
    │ --host STR        (default: 0.0.0.0)              │
    │ --port INT        (default: 8080)                 │
    ╰───────────────────────────────────────────────────╯
