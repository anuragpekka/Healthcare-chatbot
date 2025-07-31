# 🩺 HealthCare Chatbot

## 💡 Overview

The aim of this project is to create a domain-specific chatbot capable of engaging in meaningful health-related conversations. It utilizes Retrieval-Augmented Generation (RAG) to deliver context-aware responses and leverages a powerful LLM for fluent and natural interaction.


# Steps to run
## STEP 01:

Clone the repository

```bash
Project repo: https://github.com/
```
## STEP 02- Create a conda environment after opening the repository

```bash
conda create -n medibot python=3.10 -y
```

```bash
conda activate medibot
```


## STEP 03- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone & Google credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GOOGLE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

```ini
Set configurations in config.py
```

```bash
# run the following command to store embeddings to pinecone
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

```bash
In web browser open localhost
```


### Techstack Used:

- Python
- LangChain
- Flask
- Gemini
- Pinecone
