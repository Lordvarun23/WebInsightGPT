from googlesearch import search
import requests
from bs4 import BeautifulSoup
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
import cohere
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
history =[]

#Semantic chunking model details
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
)

def  return_urls(user_query):
    """
        Queries Google to retrieve URLs related to the user query.
        Args:
            user_query (str): The query for which to search URLs.
        Returns:
            list of str: A list of URLs returned from the search.
    """
    urls = []
    for url in search(user_query):
        urls.append(url)
    return urls

def return_text_from_url(url):
    """
    Sends a GET request to the specified URL and extracts the text content.
    Args:
        url (str): The URL from which to extract text.

    Returns:
        str: The text content of the webpage if the request is successful.
        int: 404 if the request fails.
    """
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the content of the webpage
        html_content = response.text

        # Optionally, use BeautifulSoup to parse the HTML
        soup = BeautifulSoup(html_content, 'html.parser')

        text = soup.get_text(separator=' ', strip=True)

        return text
    else:
        return 404

def return_documents_splitters(text):
    """
    Splits the provided text into semantic chunks using a semantic chunking model.
    Args:
        text (str): The text to be split into chunks.
    Returns:
        list of Document: A list of documents resulting from the semantic chunking.
    """

    text_splitter = SemanticChunker(hf, breakpoint_threshold_type="percentile")
    docs = text_splitter.create_documents([text])
    return docs


def return_documents(user_query,topk=1):
    """
    Retrieves and processes documents for the given user query by fetching URLs, extracting text,
    and performing semantic chunking.
    Args:
        user_query (str): The query for which to retrieve documents.
        topk (int): The number of top URLs to process.

    Returns:
        list of Document: A list of documents obtained from the processed texts.
    """
    # Finding relevant urls for given user query from google search
    urls = return_urls(user_query)

    # extracting text from each urls
    texts = []
    i = 0
    for url in urls:
        res = return_text_from_url(url)
        if res!=404 and i<topk:
            texts.append(str(res))
            i+=1
    print(">>Extracted URLS")
    # Doing semantic chunking to get documents
    collection = []
    for text in texts:
        collection.extend(return_documents_splitters(text))
    print(">>Semantic Chunking")
    return collection

def rag_pipeline(user_query,documents):
    """
    Implements the Retrieval-Augmented Generation (RAG) pipeline to generate responses based on the provided documents.
    Args:
        user_query (str): The query for which to generate a response.
        documents (list of Document): The documents to use for generating the response.
    Returns:
        str: The generated response from the chatbot.
    """

    COHERE_API_KEY_TEXT = "YqDdRwGjdbF1N4ybPERe22CUOVzpahrVlqd3JBPs"

    index = faiss.IndexFlatL2(len(hf.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=hf,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)
    print(">>Added to FAISS")

    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    context = [i.page_content for i in retriever.invoke(user_query)]
    print(">>Retrieved context")

    if len(history)<3:
        prompt = f"""
            You are a chatbot built for helping users queries.
            Answer the question:{user_query} only based on the context: {context} provided.
            Try to answer in bulletin points.
            Give easy to understand responses.
            Do not divulge any other details other than query or context.
            If the question asked is a generic question or causal question answer them without using the context.
            If the question is a general question, try to interact with the user in a polite way. """
    else:
        prompt = f"""
        You are a chatbot built for helping users queries.
        Answer the question:{user_query} only based on the context: {context} chat history:{history[len(history)-3:]} provided.
        Try to answer in bulletin points.
        Give easy to understand responses.
        Do not divulge any other details other than query or context.
        If the question asked is a generic question or causal question answer them without using the context.
        If the question is a general question, try to interact with the user in a polite way. """

    co = cohere.Client(COHERE_API_KEY_TEXT)
    response = co.chat(message=prompt, model="command-r", temperature=0)
    print(">>Generated response")
    #print(response.text)
    history.append(response.text)
    return response.text

def chatbot_response(user_query):
    """
    Generates a response based on the user query, handling specific greetings and general queries.
    Args:
        user_query (str): The query from the user.
    Returns:
        str: The response generated by the chatbot.
    """

    documents = return_documents(user_query)
    reply = rag_pipeline(user_query, documents)
    return reply

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_input = request.json.get('message')
    response = chatbot_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)