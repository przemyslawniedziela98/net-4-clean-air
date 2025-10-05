# Net4CleanAir Literature Review Explorer

**Project Description:**  
This application supports the exploration of literature and market review results from **CA23139 - Network for Indoor Air Cleaning (Net4CleanAir)**.  

The Net4CleanAir project aims to consolidate accessible information on air cleaning technologies and create an international, interdisciplinary network of experts and stakeholders. 

**Application Purpose:**  
The app allows users to query and explore literature review results, providing embeddings-based search across indexed papers. Users can quickly find papers by topics, aims, findings, and other metadata, presented in an interactive interface.

**Technology Stack:**  
Python, Flask, Qdrant (vector search), SentenceTransformers (text embeddings), OpenAI, Jinja2 + Bootstrap (UI), Docker

**Local Setup**
Make sure that 
1. Clone the repo:
   ```bash
   git clone https://github.com/przemyslawniedziela98/net-4-clean-air

2. Set as env variable OpenAI API key:
    ```bash
   export OPENAI_API_KEY="your_api_key"

3. Run Qdrant via Docker:
Make sure you have Docker installed and running.
    ```bash
    ./run_docker.sh

4. Run Flask app:
    ```bash
    ./run_flask.sh


