# AI Discussion Studio

AI Discussion Studio is a local FastAPI application for running multi-participant AI discussions from configuration.

The project is built around three ideas:

- participant-centric setup: each participant directly declares `name`, `model`, `prompt`, and optional `role_label`
- turn-based dialog: every later participant sees the full transcript and can support, rebut, revise, or synthesize
- config-first workflows: templates, models, and protocol profiles all come from JSON and prompt files instead of hard-coded role registries

## What The App Does

The app provides a browser UI where you can:

- choose a discussion template
- load that template's default participants
- adjust models, prompts, and runtime settings
- run a live discussion and watch turns appear chronologically
- review previous sessions

The current template library focuses on discussion styles rather than raw model benches:

- `Starter Dialog`
- `Debate Arena`
- `Brainstorm Studio`
- `Werewolf Table`

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
│       ├── general/
│       ├── debate/
│       ├── brainstorm/
│       └── werewolf/
├── data/
├── .env.example
└── requirements.txt
```

## Configuration Model

### `config/providers.json`

Connection layer only.

Each provider entry stores protocol and request information such as:

- `provider_id`
- `display_name`
- `protocol_adapter`
- `base_url`
- `api_key_env_var`

This project intentionally keeps only two protocol-level providers:

- `openai_compatible`
- `anthropic_compatible`

### `config/models.json`

UI model catalog only.

This file controls which models appear in the dropdowns. It does not drive discussion logic by itself.

Each model entry includes:

- `model`
- `label`
- `provider_id`
- `enabled`

### `config/discussions.json`

Primary runtime source of truth for discussion behavior.

Each template defines:

- `discussion_id`
- `display_name`
- `description`
- `max_turn_cycles`
- `participants`

Each participant defines:

- `name`
- `model`
- `prompt`
- optional `role_label`
- optional `enabled`

## Prompt Layout

Prompts are grouped by discussion style:

- `prompts/dialog/general/`
  baseline answer / alternative / critique / synthesis
- `prompts/dialog/debate/`
  structured debate roles such as proposition, opposition, cross-examiner, judge
- `prompts/dialog/brainstorm/`
  framing, wildcard ideation, builder, risk filter, curator
- `prompts/dialog/werewolf/`
  narrator, hidden wolf players, seer, witch, hunter, villager, vote counter

Prompt references use `file:` paths, for example:

```text
file:dialog/general/answerer.txt
file:dialog/debate/debate_judge.txt
file:dialog/werewolf/werewolf_vote_counter.txt
```

## Runtime Behavior

The orchestrator runs discussions as chronological turns:

1. load the selected template
2. resolve enabled participants in order
3. run one turn per participant per cycle
4. append each response to `session.turns[]`
5. let the synthesizer-style participant produce the final consolidated answer

Later turns always receive the full transcript so far. This makes the flow feel like a real conversation rather than isolated parallel completions.

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

Use the sanitized backup file already included in the repo:

```bash
cp .env.example .env
```

Then fill in your own key:

```dotenv
DASHSCOPE_CODING_API_KEY=your_real_key_here
```

### 4. Run the app

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Security And Commit Hygiene

- `.env` is ignored and should never be committed
- `.env.example` is the sanitized backup intended for version control
- session history under `data/sessions.json` is ignored
- Python caches and generated image files are ignored

Before sharing or committing, make sure:

- no real API key remains in tracked files
- session history has been cleared if it contains private prompts or outputs
- only sanitized configuration templates are staged

## Recommended First Edits

If you want to customize the app without touching core code, the highest-value files are:

- `config/discussions.json`
- `config/models.json`
- `config/providers.json`
- `prompts/dialog/...`

## Stack

- FastAPI
- Jinja2 templates
- lightweight front-end JavaScript
- KaTeX for math rendering
- protocol adapters for OpenAI-compatible and Anthropic-compatible APIs
