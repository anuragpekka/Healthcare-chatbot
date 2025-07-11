from flask import Flask, render_template, request
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import logging
from logging.handlers import RotatingFileHandler
import markdown

from src.helper import download_google_embeddings
from src.prompt import system_prompt_concise
import config

app = Flask(__name__)

load_dotenv()

# Configure rotating file handler
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler = RotatingFileHandler(config.LOG_FILE, maxBytes=config.LOG_MAXBYTES, backupCount=config.LOG_BACKUP_COUNT)
log_handler.setFormatter(log_formatter)

logger = logging.getLogger(__name__)
logger.addHandler(log_handler)
logger.setLevel(config.LOG_LEVEL)

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')

embeddings = download_google_embeddings(GOOGLE_API_KEY)

index_name = config.INDEX_NAME

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type=config.VECTOR_STORE_SEARCH_TYPE, search_kwargs={"k":3})

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(
    model=config.MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=config.LLM_TEMPERATURE,
    max_output_tokens=config.LLM_MAX_TOKENS
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", config.SYSTEM_PROMPT),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# For short answer
prompt_short = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_concise),
            ("human", "{input}")
        ]
)
question_short_answer_chain = create_stuff_documents_chain(llm, prompt_short)
rag_chain_short = create_retrieval_chain(retriever, question_short_answer_chain)

rag_chain.invoke({"input": "Hi"})
rag_chain_short.invoke({"input":"Hi"})

@app.route("/", methods={"GET"})
def index():
    logger.info("Reached the the Home page")
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    logger.debug(f"Input message: {msg}")
    if("submit_type" in request.form):
        if request.form["submit_type"] == "short_answer":
            logger.info("Fetching short response.")
            response = rag_chain_short.invoke({"input": msg})
            logger.debug(f"Response : {response['answer']}")
            return str(response["answer"])

    logger.info("Fetching long response.")
    response = rag_chain.invoke({"input": msg})
    logger.debug(f"Response : {response['answer']}")
    return markdown.markdown(str(response["answer"]))
    
    
if __name__ == '__main__':
    app.run(host=config.HOST_IP, port=config.HOST_PORT, debug=True)
