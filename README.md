# Net4CleanAir Literature Review Explorer

**Project Description:**  
This application supports the exploration of literature and market review results from **CA23139 - Network for Indoor Air Cleaning (Net4CleanAir)**.  

The Net4CleanAir project aims to consolidate accessible information on air cleaning technologies and create an international, interdisciplinary network of experts and stakeholders. 
The app includes monitoring of performance  metrics such as search latency, number of searches, and errors. Metrics are exposed to Prometheus and can be visualized in Grafana inside UI. Additional latency experiments can be found in the `experiments` folder.


**Application Purpose:**  
The app allows users to query and explore literature review results, providing embeddings-based search across indexed papers. Users can quickly find papers by topics, aims, findings, and other metadata, presented in an interactive interface.

**Technology Stack:**  
Python, Flask, Qdrant (vector search), SentenceTransformers (text embeddings), OpenAI, Jinja2 + Bootstrap (UI), Docker, Prometheus, Grafana

---
**Local Setup**
Local Development Setup

### 1. Clone the repository

```bash
git clone https://github.com/przemyslawniedziela98/net-4-clean-air.git
cd net-4-clean-air
```

Alternatively, download it via GitHub web interface:  
[https://github.com/przemyslawniedziela98/net-4-clean-air](https://github.com/przemyslawniedziela98/net-4-clean-air)

---

### 2. Set your OpenAI API Key

You need a valid OpenAI API key to generate embeddings and run semantic search.  
If you don’t have one, you can create it here: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)

#### macOS / Linux (bash/zsh):

```bash
export OPENAI_API_KEY="your_api_key"
```

To make this persistent, add it to your `~/.bashrc` or `~/.zshrc`.

#### Windows (PowerShell):

```powershell
setx OPENAI_API_KEY "your_api_key"
```

Then restart PowerShell so the variable is available in the new session.

### 3. Run Qdrant (Vector DB), Prometheus and Grafana via Docker

Make sure [Docker](https://docs.docker.com/get-docker/) is installed and running.

Start Qdrant locally using the provided script:

#### macOS / Linux:
```bash
./run_docker.sh
```

This will start a local Qdrant server (by default on port `6333`) as well as Prometheus (port `9090`) and Grafana (port `3000`).
For local setup on Windows, it is necessary to use [Git Bash](https://gitforwindows.org/) or manually execute the commands from the Bash scripts in PowerShell.

---

### 4. Run the Flask Application

The repository includes a helper script **`run_flask.sh`**, which automatically sets up a Python virtual environment, installs all `pip` dependencies from `requirements.txt`, and launches the Flask app.

#### macOS / Linux:

```bash
./run_flask.sh
```

If you have [Git Bash](https://gitforwindows.org/) installed, you can run `./run_flask.sh` directly on Windows.

### 5. Accessing the App

Once Flask starts, open your browser and go to:

[http://localhost:5000](http://localhost:5001)

You should see the Net4CleanAir interface. <br>
Additionally, you can view monitoring dashboards via Grafana at http://localhost:3000.

---

## Notes

- The `run_flask.sh` script is idempotent – it can be safely re-run. It checks for a virtual environment and installs dependencies as needed.  
- All configuration is currently done via environment variables (no `.env` file is required).  
- Qdrant must be running in the background for search to work.
- Tests can be run using the provided test script `run_tests.sh`.

---

### Useful Links

- Qdrant Documentation: [https://qdrant.tech/documentation/](https://qdrant.tech/documentation/)  
- OpenAI Embeddings Guide: [https://platform.openai.com/docs/guides/embeddings](https://platform.openai.com/docs/guides/embeddings)  
- SentenceTransformers: [https://www.sbert.net/](https://www.sbert.net/)