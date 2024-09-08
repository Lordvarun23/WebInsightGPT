# WebInsightGPT
 WebInsightGPT is a real-time chatbot that queries the web, scrapes top sites, and performs semantic chunking to store data in a vector database. It retrieves relevant chunks to generate accurate, context-aware responses, ensuring up-to-date and precise information for users.

## Features

- **Real-Time Web Queries:** Fetches top websites in response to user queries.
- **Content Extraction:** Scrapes and extracts text from web pages.
- **Semantic Chunking:** Processes and chunks content to enhance retrieval accuracy.
- **Vector Database Integration:** Utilizes FAISS for efficient vector-based retrieval.
- **Context-Aware Responses:** Generates answers based on the most relevant context and user history.
- **Conversational Interaction:** Handles casual and general questions with appropriate responses.

## Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/Lordvarun23/webinsightgpt.git](https://github.com/Lordvarun23/WebInsightGPT)
   cd webinsightgpt
   ```
2. Create a virtual environment and activate it:
 ```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install the required packages:
```
pip install -r requirements.txt
```

## Usage
1. Start the Flask application:
```
python main.py
```
2. Open your browser and navigate to http://localhost:5000 to access the chatbot interface.
3. Interact with the chatbot by typing your queries. The bot will fetch information from the web, process it, and provide responses based on the context.

Code Overview
- app.py: Main application file that sets up the Flask server, defines routes, and processes user queries.
- return_urls(user_query): Queries Google for relevant URLs based on the user query.
- return_text_from_url(url): Scrapes text content from the given URL.
- return_documents_splitters(text): Performs semantic chunking on the extracted text.
- return_documents(user_query, topk=1): Retrieves and processes documents for the given query.
- rag_pipeline(user_query, documents): Implements the RAG pipeline using FAISS and generates responses with Cohere's API.
- chatbot_response(user_query): Handles user queries and provides responses based on context or general interactions.

## Try 
![image](https://github.com/user-attachments/assets/7045ce48-1993-44d9-9763-1b5a2a69e15e)
