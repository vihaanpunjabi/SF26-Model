# Repo guide for AI coding agents

Summary
- This repository currently contains a single file: `model.py` (currently empty).
- There are no README, dependency manifests, tests, or CI files present.

What to prioritize
- When making changes, focus on `model.py` as the primary implementation artifact.
- If you add runtime dependencies, create `requirements.txt` at the repo root.
- Add a `README.md` describing purpose and run instructions when you introduce features.

Concrete, discoverable patterns (from this repo)
- File layout: single top-level Python module: `model.py`.
- No package directory or tests exist; treat changes as single-file Python edits unless you add more modules.

Agent workflows and examples
- Running code: there is no build system; run with `python model.py` from the repo root.
- Adding deps: add `requirements.txt` and instruct the user to create a venv: `python -m venv .venv` then `source .venv/bin/activate` and `pip install -r requirements.txt`.
- Testing convention (recommended explicit example): create `tests/` and use `pytest`. Example command: `pytest -q`.

Coding conventions and expectations
- Keep changes minimal and focused: single-file edits are preferred unless adding clear supporting modules.
- If creating new modules, add an `__init__.py` and update `requirements.txt` as needed.

Integration points & external dependencies
- None are present. If you add integrations (APIs, databases, model artifacts), document them in `README.md` and add config files at the repo root.

What not to assume
- Do not assume any CI, test runners, or packaging are present — create them and document their usage if you add them.

When to ask the user
- If a design decision affects repository layout (package vs single-file), or if you need credentials/config to run integrations, stop and ask.

Notes for maintainers
- This file was generated from an initial repository scan. If you want richer agent guidance, add a short README, example run commands, and any existing dev scripts.

Questions for you
- Do you want me to initialize basic tooling (README, requirements.txt, tests/) now?
