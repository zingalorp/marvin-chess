#!/bin/bash
# Move to the engine directory (where this script lives)
cd "$(dirname "$0")"

# Run the engine using the local venv and unbuffered output
exec ./venv/bin/python -u -m inference.uci_engine "$@"