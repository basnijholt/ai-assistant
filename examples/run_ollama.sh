#!/bin/sh
ollama serve &
PID=$!
sleep 5
ollama pull qwen3:4b
wait $PID
