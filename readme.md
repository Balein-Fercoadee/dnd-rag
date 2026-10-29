# Local D&D 3.5 Rules RAG Service

This project indexes text and tables extracted from PDF rulebooks in Qdrant,
uses Ollama for embeddings and local chat generation, and provides an
OpenAI-compatible API for a Home Assistant voice assistant.

The expected layout on the Ubuntu host is:

```text
~/dnd-rag/
├── app.py
├── ingest.py
├── rag/
│   ├── __init__.py
│   ├── rag_config.py
│   ├── rag_retrieval.py
│   └── rag_rules.py
├── extraction/
│   ├── __init__.py
│   └── table_extraction.py
├── requirements-rag-service.txt
└── pdfs/
    └── *.pdf
```

Keep the PDFs and Qdrant data on storage you control. This setup sends prompts
and extracted text only to the local Ollama and Qdrant services.

## Components

- **Ollama** serves the chat model and embedding model.
- **Qdrant** stores PDF chunks and normalized table rows.
- **FastAPI/Uvicorn** provides `/v1/chat/completions` and `/v1/models`.
- **Home Assistant Green** connects to the API using an OpenAI-compatible
  conversation integration, such as Extended OpenAI Conversation.

The service defaults to `qwen3:8b`, `nomic-embed-text`, Qdrant collection
`dnd35`, and local Ollama/Qdrant endpoints. Change these with the environment
variables documented below.

## 1. Install the local services

Install Ollama and Docker on Ubuntu if they are not already installed. Start
Ollama, then download the models:

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

Run Qdrant with persistent storage and its HTTP port bound to localhost. For
example:

```bash
mkdir -p "$HOME/dnd-rag/qdrant-storage"
docker run -d \
  --name dnd-qdrant \
  --restart unless-stopped \
  -p 127.0.0.1:6333:6333 \
  -v "$HOME/dnd-rag/qdrant-storage:/qdrant/storage" \
  qdrant/qdrant
```

The importer and API run on the Ubuntu host, so Qdrant does not need to be
exposed to the LAN. Keep the Qdrant storage directory persistent; deleting it
removes the collection and its indexed data.

Check the local services:

```bash
curl http://127.0.0.1:11434/api/tags
curl http://127.0.0.1:6333/healthz
```

## 2. Prepare the Python service

Copy `app.py`, `ingest.py`, `rag/`, `extraction/`, and
`requirements-rag-service.txt` into `~/dnd-rag`. Create the PDF directory and a
virtual environment:

```bash
cd "$HOME/dnd-rag"
mkdir -p pdfs
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-rag-service.txt
```

Place the PDFs to index directly in `~/dnd-rag/pdfs/`. The default importer
searches for `*.pdf` in that directory (not recursively in subdirectories).

## 3. Configure the service

Both Python programs read configuration from environment variables. The
defaults are suitable when Ollama, Qdrant, the importer, and the API all run on
the same Ubuntu machine:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PDF_DIR` | `~/dnd-rag/pdfs` | Directory of PDFs for ingestion |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama base URL |
| `QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant base URL |
| `CHAT_MODEL` | `qwen3:8b` | Ollama model used to answer questions |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama model used for vectors |
| `QDRANT_COLLECTION` | `dnd35` | Qdrant collection name |
| `RAG_API_KEY` | unset | Required API key for `/v1/models` and chat |
| `RETRIEVAL_LIMIT` | `8` | Number of semantic results to retrieve |
| `BONUS_ROW_LIMIT` | `2` | Maximum matching class/bonus rows per source lookup |

Choose a long, private API key and keep it out of source control. For a quick
interactive run, export the variables in the shell before starting the API.
For a persistent systemd service, create a root-readable environment file:

```bash
sudo install -m 600 /dev/null /etc/dnd-rag.env
sudoedit /etc/dnd-rag.env
```

Example contents (replace the key with a secret of your own):

```ini
RAG_API_KEY=replace-with-a-long-random-secret
PDF_DIR=/home/YOUR_USER/dnd-rag/pdfs
OLLAMA_URL=http://127.0.0.1:11434
QDRANT_URL=http://127.0.0.1:6333
CHAT_MODEL=qwen3:8b
EMBED_MODEL=nomic-embed-text
QDRANT_COLLECTION=dnd35
```

Use the same API key in Home Assistant. Do not paste a real key into a
published README, chat, or source repository.

## 4. Preview and ingest PDFs

Preview extraction from a specific PDF page before indexing it. Page numbers
are **PDF page numbers**, which may differ from printed page numbers:

```bash
cd "$HOME/dnd-rag"
. .venv/bin/activate
python ingest.py --pdf "pdfs/Player's Handbook.pdf" --preview-page 56
```

Ingest all PDFs directly in `PDF_DIR`:

```bash
python ingest.py
```

Or ingest just one PDF:

```bash
python ingest.py --pdf "pdfs/Player's Handbook.pdf"
```

The importer extracts page text in detected column order, captioned tables,
generic table rows with caption/header context, normalized class spell-per-day
rows, and ability-score bonus-spell rows. Column-aware page text keeps nearby
rules in separate columns from being merged into one passage.

### PDF layout and retrieval lessons

The Player's Handbook spell pages use **three columns**. Do not assume a PDF
uses only two: the importer detects vertical whitespace gutters and extracts
each detected column separately, preserving their reading order. If no clear
gutters are detected, it falls back to extracting the page as a whole. This
prevents text from neighboring spells (for example, their different damage
caps) from being combined into one retrieval passage. When a result looks
wrong, use `--preview-page` on the source page to check what the PDF extractor
actually produced before changing answer logic.

Page text and table extraction serve different purposes. Page text is
column-aware, but table detection still uses the full-page extracted layout;
adjacent side-by-side tables or merged cells may therefore remain searchable
in page text without becoming separate row records. Generic table rows preserve
extracted cell text without guessing column meanings, while full table and page
text remain indexed as context.

The extracted records are embedded with Ollama and upserted into the configured
Qdrant collection. Before reindexing a PDF, the importer removes previously
indexed points with that PDF's filename; therefore keep filenames unique across
books. Changes to extraction or chunking only affect the index after reingesting
the PDFs. Changes limited to API answer logic do not require reingestion.

Class tables are indexed both as complete captioned table blocks and as
individual normalized spell rows. Each normalized row also carries its
original extracted row text, retaining columns such as attack bonuses, saves,
special features, and spell counts in the row's searchable text.
Casting abilities are attached only when the extracted text contains a
class-specific rule linking that class and ability; ambiguous nearby text
does not establish the ability.
Class progression tables also get structured per-level base attack bonus rows,
which let the API add BAB contributions across multiclass characters instead
of selecting only the highest-level class. The same indexed rows include
Fortitude, Reflex, and Will base-save contributions, which the API sums
independently for multiclass saving-throw questions.
For multiclass spell questions, the API reports each class's spells-per-day
progression separately and applies bonus spells only from that class's
spellcasting ability when the relevant ability score is provided.

Watch for a final `Done. Indexed ...` message. If a class row is not
normalized, preview its PDF page and inspect the extracted table heading and
rows. Scanned/image-only PDFs may need OCR before this text-based ingestion
can extract their contents.

In the code, `ingest.py` handles PDF collection, embedding, and Qdrant
updates; `extraction/table_extraction.py` contains table detection and row
normalization.

## 5. Run the API

The API is organized by responsibility: `app.py` exposes the FastAPI endpoints,
and the `rag/` package contains the supporting modules. `rag_config.py` holds
runtime settings and question patterns. `rag_retrieval.py` and `rag_rules.py`
preserve the existing public imports as small facades. Retrieval is separated
into `retrieval_backend.py` (Ollama embeddings and the shared Qdrant client),
`retrieval_tables.py` (structured class and ability rows), and
`retrieval_pipeline.py` (question-aware retrieval). Rule logic is separated
into `rule_parsing.py`, `rule_character.py`, `rule_combat.py`, and
`rule_spells.py`.

For a temporary foreground run:

```bash
cd "$HOME/dnd-rag"
. .venv/bin/activate
set -a
. /etc/dnd-rag.env
set +a
uvicorn app:app --host 0.0.0.0 --port 8000
```

To run it under systemd, create `/etc/systemd/system/dnd-rag.service` and
replace `YOUR_USER` with the Ubuntu account that owns the project:

```ini
[Unit]
Description=Local D&D Rules RAG API
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER/dnd-rag
EnvironmentFile=/etc/dnd-rag.env
ExecStart=/home/YOUR_USER/dnd-rag/.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now dnd-rag
sudo systemctl status dnd-rag
sudo journalctl -u dnd-rag -f
```

The API binds to all interfaces so Home Assistant can reach it over the LAN.
Allow inbound TCP port `8000` only from trusted LAN devices (or restrict it to
the Home Assistant host in the firewall). Do not forward this port from the
internet. Qdrant remains bound to localhost.

Check service health on Ubuntu:

```bash
curl http://127.0.0.1:8000/health
```

The expected response is JSON with `"status": "ok"` and the collection name.
The health endpoint does not check model generation; a chat request is needed
to test the full path.

## 6. Connect Home Assistant

Install and configure an OpenAI-compatible conversation integration in Home
Assistant, such as Extended OpenAI Conversation. Set:

- **Base URL:** `http://UBUNTU_LAN_IP:8000/v1`
- **API key:** the same value as `RAG_API_KEY`
- **Model:** `qwen3:8b`

Use the Ubuntu machine's LAN IP in place of `UBUNTU_LAN_IP`; do not use
`127.0.0.1` from Home Assistant, because that would refer to Home Assistant
itself. Confirm that Home Assistant can reach the Ubuntu host on TCP port
`8000`.

## 7. Test spell-table answers

Try questions for different classes, for example:

- “How many spells does a 5th-level cleric get?”
- “How many 5th-level spells does an 11th-level wizard get?”
- “How many 5th-level spells does an 11th-level wizard with Intelligence 20 get?”
- “What is the BAB of a barbarian 3/fighter 4 character?”

The API distinguishes base class-table slots from bonus spells and only adds
bonus spells when it can identify the class's casting ability and the matching
ability score. If extracted class metadata does not establish the ability, it
should not guess or add a bonus.

### Automated tests

Install the development test dependencies in the virtual environment:

```bash
cd "$HOME/dnd-rag"
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Format all Python files in the project with Black:

```bash
python -m black .
```

Check formatting without modifying files:

```bash
python -m black --check .
```

Run the offline tests:

```bash
python -m pytest -m "not live"
```

These tests submit sample questions through the FastAPI chat endpoint while
providing deterministic retrieved passages. They check answer calculations
and class/ability handling without contacting Ollama or Qdrant.

To exercise the running service, ingest the PDFs, start the API, then set the
API's `/v1` base URL and API key in the test shell:

```bash
export RAG_TEST_BASE_URL="http://127.0.0.1:8000/v1"
read -rsp "RAG API key: " RAG_TEST_API_KEY
export RAG_TEST_API_KEY
printf '\n'
python -m pytest -m live
unset RAG_TEST_API_KEY
```

Live tests send the sample spell questions to the configured API and check
that the responses contain expected spell counts. They validate retrieval,
indexing, and the response path together; they require a running, authenticated
service and successfully ingested rulebooks.

## Updating code or PDFs

After changing the importer, copy the updated `ingest.py` to Ubuntu and
reingest the affected PDFs. After changing the API, copy the updated `app.py`
and restart the service:

```bash
sudo systemctl restart dnd-rag
sudo journalctl -u dnd-rag -n 100 --no-pager
```

Reingest a PDF when importer logic or its extracted metadata changes; simply
restarting the API does not rebuild existing Qdrant points.
