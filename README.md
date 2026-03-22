# AI Discussion Studio

AI Discussion Studio is a local FastAPI app for running multi-participant AI discussions from editable configuration.

It is built around three ideas:

- config-first setup: providers, models, templates, and prompts live in JSON or prompt files
- participant-centric execution: each participant has its own name, model, prompt, and optional role label
- live workbench UX: the browser UI lets you start a run, watch turns arrive in order, and review recent sessions

## What You Can Do

- choose a discussion template
- load template defaults into the setup form
- adjust participants, prompts, models, and runtime settings
- run a live multi-agent discussion
- inspect the full turn timeline and final answer in the Discussion Workbench
- reopen or delete recent sessions

Current templates include:

- `Starter Dialog`
- `Debate Arena`
- `Brainstorm Studio`
- `Werewolf Table`

The werewolf mode is treated like a first-class discussion template, but its detailed rules live in code and prompts rather than in this README.

## Project Structure

```text
.
├── app/
│   ├── main.py
│   ├── config.py
│   ├── providers/
│   ├── routes/
│   ├── services/
│   ├── static/
│   └── templates/
├── config/
│   ├── providers.json
│   ├── models.json
│   └── discussions.json
├── prompts/
│   └── dialog/
├── data/
├── .env.example
└── requirements.txt
```

## Configuration

### `config/providers.json`

Defines protocol-level providers and request routing details such as:

- `provider_id`
- `display_name`
- `protocol_adapter`
- `base_url`
- `api_key_env_var`

### `config/models.json`

Defines the model catalog shown in the UI. Each entry includes:

- `model`
- `label`
- `provider_id`
- `enabled`

### `config/discussions.json`

Defines runtime templates. Each template includes:

- `discussion_id`
- `display_name`
- `description`
- `max_turn_cycles`
- `participants`

Each participant includes:

- `name`
- `model`
- `prompt`
- optional `role_label`
- optional `enabled`

## Prompts

Prompt references use `file:` paths, for example:

```text
file:dialog/general/answerer.txt
file:dialog/debate/debate_judge.txt
file:dialog/werewolf/werewolf_vote_counter.txt
```

Prompt folders are grouped by discussion style:

- `prompts/dialog/general/`
- `prompts/dialog/debate/`
- `prompts/dialog/brainstorm/`
- `prompts/dialog/werewolf/`

## Runtime Notes

- standard discussion templates run in chronological turn order
- later turns can see the prior transcript
- werewolf mode uses a dedicated game engine with public state and moderator-only state
- session results are stored locally under `data/`

## Local Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create a local `.env`

```bash
cp .env.example .env
```

Then fill in your own API key locally.

### 4. Run the app

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Security

- `.env` is ignored and should never be committed
- `.env.example` is the sanitized template for version control
- `data/*.json` is ignored so local session history and test runs stay out of commits
- never leave real API keys in tracked files, screenshots, or copied examples

## Recommended First Edits

- `config/discussions.json`
- `config/models.json`
- `config/providers.json`
- `prompts/dialog/...`

## Stack

- FastAPI
- Jinja2 templates
- lightweight front-end JavaScript
- KaTeX for math rendering
- OpenAI-compatible and Anthropic-compatible protocol adapters
