---
description: Set up Nur to call you (your number, your name, your EESI API key)
argument-hint: +14155550123 [your name]
allowed-tools: Bash(eesi companion setup:*), Bash(eesi companion confirm:*), Bash(eesi companion status:*), Bash(python3 -m eesi companion:*)
---

Set up Nur's coding companion for me with the Bash tool. Use `eesi companion ...`, or `python3 -m eesi companion ...` if `eesi` is not found; if neither works, tell me to install the SDK with `uv tool install git+https://github.com/eesi-ai/eesi-sdk-python` and stop.

1. Run `eesi companion setup --phone <number> --name <name>` with the number and name in: $ARGUMENTS. Ask me for my number (E.164, like +14155550123) if there is none.
2. If it says the API key is missing, do not ask me to paste the key here: tell me to run `eesi companion setup --api-key sk-eesi-...` in my own terminal (or set EESI_API_KEY), then continue.
3. Nur then phones my number with a six-digit code. Ask me for the code and run `eesi companion confirm <code>`.
4. Run `eesi companion status` and tell me in one sentence whether Nur is ready to call me.
