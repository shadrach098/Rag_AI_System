---

# Project Documentation: RAG AI System

## Table of Contents

1.  [Introduction](#1-introduction)
2.  [Project Overview](#2-project-overview)
    *   [Core Functionality](#21-core-functionality)
    *   [Key Technologies](#22-key-technologies)
3.  [System Architecture](#3-system-architecture)
    *   [High-Level Diagram](#31-high-level-diagram)
    *   [Component Breakdown](#32-component-breakdown)
        *   [1. Document Processing and Embedding Layer](#321-document-processing-and-embedding-layer)
        *   [2. Vector Store Management Layer](#322-vector-store-management-layer)
        *   [3. API Endpoints (Flask Server)](#323-api-endpoints-flask-server)
        *   [4. Retrieval-Augmented Generation (RAG) Layer](#324-retrieval-augmented-generation-rag-layer)
        *   [5. User Interface (Streamlit GUI)](#325-user-interface-streamlit-gui)
4.  [Technical Details and Implementation](#4-technical-details-and-implementation)
    *   [Embedding Functions (`Encrypy` and `Qdrant_Encrypy`)](#41-embedding-functions-encrypy-and-qdrant_encrypy)
    *   [Retrieval Function (`Qdrant_Retrival`)](#42-retrieval-function-qdrant_retrival)
    *   [Flask API (`/textembeddings/faiss` and `/VecStore`)](#43-flask-api-textembeddingsfaiss-and-vecstore)
    *   [Streamlit GUI](#44-streamlit-gui)
    *   [Error Handling](#45-error-handling)
5.  [Setup and Installation](#5-setup-and-installation)
    *   [Prerequisites](#51-prerequisites)
    *   [1. Clone the Repository](#52-1-clone-the-repository)
    *   [2. Install Python Dependencies](#53-2-install-python-dependencies)
    *   [3. Docker Setup for Qdrant (Mandatory for Qdrant Backend)](#54-3-docker-setup-for-qdrant-mandatory-for-qdrant-backend)
    *   [4. Set up OpenAI API Key](#55-4-set-up-openai-api-key)
    *   [5. Run the Application](#56-5-run-the-application)
6.  [Usage Guide](#6-usage-guide)
    *   [1. Launching the Application](#61-1-launching-the-application)
    *   [2. Uploading a Document](#62-2-uploading-a-document)
    *   [3. Interacting with the Document](#63-3-interacting-with-the-document)
7.  [Features](#7-features)
8.  [Troubleshooting](#8-troubleshooting)
9.  [Future Enhancements](#9-future-enhancements)
10. [Contact](#10-contact)

---

## 1. Introduction

The **RAG AI System** is a robust and interactive application designed to facilitate efficient document understanding and retrieval-augmented generation (RAG). It enables users to upload various document types (PDF, CSV, DOCX), process them into vectorized embeddings, store them in scalable vector databases (Qdrant or FAISS), and then interactively query these documents using natural language. The system leverages state-of-the-art Large Language Models (LLMs) and vector search capabilities to provide contextually accurate answers directly from the uploaded content.

This project addresses the challenge of extracting specific information from large, unstructured documents without the need for manual parsing, making it invaluable for knowledge base querying, research, and customer support applications.

## 2. Project Overview

### 2.1 Core Functionality

*   **Document Ingestion**: Supports uploading and processing of PDF, CSV, and DOCX files.
*   **Text Extraction**: Extracts plain text from various document formats.
*   **Text Chunking**: Divides large documents into manageable chunks for effective embedding.
*   **Vector Embeddings**: Generates high-dimensional numerical representations (embeddings) of text chunks using OpenAI's embedding models.
*   **Vector Storage**: Provides a choice between two popular vector database backends:
    *   **Qdrant**: A high-performance vector similarity search engine, suitable for large-scale production deployments.
    *   **FAISS**: (Facebook AI Similarity Search) A library for efficient similarity search and clustering of dense vectors, ideal for local and smaller-scale deployments.
*   **Retrieval-Augmented Generation (RAG)**: Integrates document retrieval with a conversational AI model. When a user asks a question, the system retrieves the most relevant document chunks and uses them as context for the LLM to generate an informed answer.
*   **Conversational AI Interface**: A user-friendly Streamlit interface for natural language interaction.
*   **Image-to-Text (OCR)**: Basic OCR capability for extracting text from images submitted in the chat interface.

### 2.2 Key Technologies

*   **Python**: Primary programming language.
*   **LangChain**: Framework for building LLM applications, used for text loading, splitting, embeddings, prompt templating, and RAG pipeline construction.
*   **OpenAI**: Provides embedding models and the underlying LLM for generation.
*   **Qdrant**: Vector database for efficient similarity search.
*   **FAISS**: Local vector store for efficient similarity search.
*   **Flask**: Lightweight web framework for building the backend API.
*   **Streamlit**: Framework for creating interactive web applications (the GUI).
*   **PyPDF2**: For PDF document parsing.
*   **pandas**: For CSV document processing.
*   **python-docx**: For DOCX document parsing.
*   **Pillow (PIL)** & **pytesseract**: For image processing and Optical Character Recognition (OCR).
*   **requests**: For making HTTP requests to the Flask API.
*   **Docker**: For containerizing and running the Qdrant vector database.

## 3. System Architecture

### 3.1 High-Level Diagram

```
+----------------+      HTTP/API Call      +---------------+      Function Calls      +--------------------------+
|                | <---------------------> |               | <----------------------> |                          |
| Streamlit GUI  |                         |  Flask API    |                          | Backend Python Functions |
| (User Interface)|                         | (Web Server)  |                          | (Logic & Orchestration)  |
|                |                         |               |                          |                          |
+----------------+                         +---------------+                          +--------------------------+
        ^                                          ^                                             |
        | Document Upload                          | Query/Response                              |
        |                                          |                                             | OpenAI
        v                                          v                                             | Embeddings
+---------------------+                    +---------------------+                    +----------------+
| Document Storage    |                    | Vector Stores       | <-----------------> | OpenAI API     |
| (Endpoint_Files/)   |                    | (Qdrant / FAISS)    |                     | (Embeddings, LLM) |
+---------------------+                    +---------------------+                     +----------------+
                                                 (Qdrant via Docker)
```

### 3.2 Component Breakdown

#### 3.2.1. Document Processing and Embedding Layer

This layer is responsible for handling raw document inputs and converting them into a format suitable for vector storage.

*   **Input**: Raw documents (PDF, CSV, DOCX).
*   **TextLoader**: Loads text content from various file types.
*   **RecursiveCharacterTextSplitter (`CH`)**: Splits the loaded text into smaller, overlapping chunks. This is crucial for maintaining context within chunks while keeping them small enough for embedding models.
*   **OpenAIEmbeddings**: Transforms text chunks into high-dimensional numerical vectors (embeddings). These embeddings capture the semantic meaning of the text.

#### 3.2.2. Vector Store Management Layer

This layer manages the persistence and retrieval of the generated text embeddings.

*   **QdrantClient**: Interacts with the Qdrant vector database (running locally via Docker). It handles collection creation, document addition, and similarity searches.
*   **FAISS**: Provides local, in-memory, or disk-persisted vector storage and similarity search capabilities.
*   **Vectorstore (`QdrantVectorStore`, `FAISS.from_documents`)**: LangChain integrations with Qdrant and FAISS, abstracting the details of adding documents and performing searches.

#### 3.2.3. API Endpoints (Flask Server)

A Flask application acts as the backend server, exposing two key API endpoints for the Streamlit GUI to interact with.

*   **`/textembeddings/faiss` (Document Ingestion & Embedding)**:
    *   **Input**: JSON payload containing `Document` content, `Name` (for the file), and `Vstore` preference (Qdrant or Faiss).
    *   **Functionality**:
        *   Saves the raw document content to a local file (`./Endpoint_Files/`).
        *   Triggers the appropriate embedding function (`Qdrant_Encrypy` or `Encrypy`) based on `Vstore`.
        *   Returns the path/collection name of the created vector store.
    *   **Error Handling**: Catches and returns errors during file operations or embedding.

*   **`/VecStore` (Similarity Search & RAG)**:
    *   **Input**: JSON payload containing `Vector` (location/collection name of the vector store) and `Query` (user's question).
    *   **Functionality**:
        *   Determines if the vector store is Qdrant or FAISS based on the `Vector` path.
        *   Loads the specified vector store.
        *   Performs a retrieval-augmented generation (RAG) operation using the `Qdrant_Retrival` function.
        *   Returns the LLM's response.
    *   **Error Handling**: Catches and returns errors during vector store loading or RAG execution.

#### 3.2.4. Retrieval-Augmented Generation (RAG) Layer

This is the core intelligence layer, responsible for generating answers based on the most relevant retrieved information.

*   **Retriever (`newDB.as_retriever()`)**: Finds the most semantically similar document chunks to a given user query from the selected vector store.
*   **Prompt Templating (`ChatPromptTemplate.from_template`)**: Structures the user's question and the retrieved context into a clear prompt for the LLM.
*   **LLM Integration (`llm`)**: Uses a Large Language Model (e.g., OpenAI's GPT models) to synthesize an answer.
*   **RunnablePassthrough**: Passes the user query directly to the LLM.
*   **StrOutputParser**: Extracts the final string output from the LLM's response.
*   **RAG Chain**: Combines the retriever, prompt template, and LLM into a seamless pipeline.

#### 3.2.5. User Interface (Streamlit GUI)

The interactive frontend for users to engage with the system.

*   **Document Upload**: Allows users to upload PDF, CSV, and DOCX files.
*   **Vector Store Selection**: A dropdown to choose between Qdrant and FAISS.
*   **Document Processing Status**: Provides real-time feedback on document processing and server communication.
*   **Chat Interface**:
    *   Users can type questions.
    *   Supports image uploads (JPG, JPEG, PNG) which are processed via OCR (pytesseract) to extract text for querying.
    *   Displays conversational history (user queries and AI responses).
    *   **Typewriter Effect**: Enhances user experience by simulating typing for AI responses.
*   **Session Management**: Uses `st.session_state` to maintain chat history and document information across interactions.

## 4. Technical Details and Implementation

### 4.1. Embedding Functions (`Encrypy` and `Qdrant_Encrypy`)

These functions handle the core logic of text splitting, embedding, and saving to the respective vector stores.

*   **`Qdrant_Encrypy(namee)`**:
    *   **Purpose**: Loads a `.txt` file, splits it into chunks, embeds the chunks, and saves them to a Qdrant collection.
    *   **Process**:
        1.  Checks if the document file exists.
        2.  Initializes a `QdrantClient` targeting `http://localhost:7000`.
        3.  Checks if a collection with `collection_name = f"{namee}_qdrant"` already exists. If yes, it returns the existing collection name, avoiding re-embedding.
        4.  If the collection doesn't exist, it creates a new Qdrant collection with `VectorParams(size=768, distance=Distance.COSINE)` (assuming 768-dimensional embeddings from `emb`).
        5.  Uses `TextLoader` to load the document from `./Endpoint_Files/{namee}.txt`.
        6.  Splits the document using `RecursiveCharacterTextSplitter` (`CH`) with `chunk_size=1000` and `chunk_overlap=100`.
        7.  Initializes `QdrantVectorStore` with the client and embedding model.
        8.  Adds the split documents (`docs`) to the Qdrant collection.
        9.  Closes the Qdrant client and returns the collection name.
    *   **Error Handling**: Includes `try-except` blocks to catch issues like Docker not running or other embedding errors, attempting to delete partially created collections on failure.

*   **`Encrypy(namee)`**:
    *   **Purpose**: Loads a `.txt` file, splits it, embeds it, and saves it to a local FAISS vector store.
    *   **Process**:
        1.  Defines a local save path `C:/Users/bruce/OneDrive/Desktop/New folder/PYTHON/Demo/Endpoint_Files/faiss/`.
        2.  Creates the directory if it doesn't exist.
        3.  Checks if a FAISS index already exists at `f'{paths}{namee}'`. If yes, it returns the existing path.
        4.  Loads the document using `TextLoader`.
        5.  Splits the document into chunks using `RecursiveCharacterTextSplitter`.
        6.  Creates a FAISS index from the documents and embeddings (`DB=FAISS.from_documents(docs,emb)`).
        7.  Saves the FAISS index locally.
        8.  Returns the local path to the saved FAISS index.

### 4.2. Retrieval Function (`Qdrant_Retrival`)

This function encapsulates the RAG pipeline. Despite its name, it handles both Qdrant and FAISS retrieval.

*   **`Qdrant_Retrival(Location, Query)`**:
    *   **Purpose**: Loads a specified vector store (Qdrant or FAISS), sets up a RAG pipeline, and performs a similarity search to answer a query.
    *   **Process**:
        1.  **Vector Store Loading**:
            *   Checks if `Location` contains "qdrant" to identify the vector store type.
            *   If Qdrant: Initializes `QdrantClient` and `QdrantVectorStore`.
            *   If FAISS: Loads the FAISS index using `FAISS.load_local` (with `allow_dangerous_deserialization=True`).
        2.  **RAG Pipeline Setup**:
            *   `QA = newDB.as_retriever()`: Converts the vector store into a retriever.
            *   `templatew`: Defines the prompt template for the LLM, instructing it to answer "only from the provided document" and respond politely if information is not found.
            *   `promptt = ChatPromptTemplate.from_template(templatew)`: Creates a LangChain chat prompt from the template.
            *   `RAG = {"context": QA, "question": RunnablePassthrough()} | promptt | llm | StrOutputParser()`: Constructs the RAG chain:
                *   The `context` is retrieved by the `QA` retriever.
                *   The `question` is passed through directly.
                *   Both are fed into the `promptt`.
                *   The prompt is sent to the `llm` (Large Language Model).
                *   The LLM's output is parsed into a string.
        3.  **Execution**: `query = RAG.invoke(Query)` executes the RAG pipeline with the user's query.
        4.  **Cleanup**: Closes the Qdrant client if it was used and explicitly `del newDB` to free up resources.
        5.  Returns the generated answer.
    *   **Error Handling**: Wraps the entire process in a `try-except` block to return specific error messages.

### 4.3. Flask API (`/textembeddings/faiss` and `/VecStore`)

The Flask application serves as the bridge between the Streamlit frontend and the backend logic.

*   **`@app.route("/VecStore",methods=["POST",'GET'])`**:
    *   Receives `Vector` (location/collection name) and `Query` from the frontend.
    *   Calls `Qdrant_Retrival` with the provided parameters.
    *   Handles potential errors from `Qdrant_Retrival`, returning a `500` status code with an error message if the call fails.
    *   Returns the RAG output if successful.
    *   Includes a `FileNotFoundError` check for FAISS paths.

*   **`@app.route("/textembeddings/faiss",methods=["POST","GET"])`**:
    *   Receives `Document` content, `Name`, and `Vstore` preference.
    *   **File Handling**:
        *   Checks if a file with `Name.txt` already exists in `Endpoint_Files/`.
        *   If it exists, it directly calls the appropriate embedding function (`Qdrant_Encrypy` or `Encrypy`).
        *   If not, it first writes the `Document` content to `Endpoint_Files/{Name}.txt`.
        *   Then calls the chosen embedding function.
    *   **Embedding & Error Handling**: Calls either `Qdrant_Encrypy` or `Encrypy` and propagates any "Error" messages with a `500` status.
    *   Returns the vector store location/collection name upon success.

### 4.4. Streamlit GUI

The `streamlit_app.py` script provides the interactive user interface.

*   **Session State Management**: Utilizes `st.session_state` to persist data like chat history (`messagechat`), uploaded document content (`message`, `cvss`, `docx`), and the active document's vector store location (`Docname`).
*   **Sidebar Controls**:
    *   `backend_option`: Dropdown for selecting Qdrant or FAISS.
    *   `Document` uploader: `st.file_uploader` for PDF, CSV, DOCX files.
    *   "Upload Document" button triggers the processing workflow.
*   **Document Upload Logic**:
    *   Reads the content based on file type (PdfReader, pandas.read_csv, docx.Document).
    *   Sends the extracted text, name, and chosen vector store to the Flask API endpoint (`/textembeddings/faiss`).
    *   Displays success or error messages from the API.
*   **Chat Input**:
    *   `st.chat_input` for user text queries and file uploads (images).
    *   **Text Queries**: Sends the query and the active `Docname` (vector store location) to the Flask API (`/VecStore`).
    *   **Image Queries**: If an image is uploaded, it uses `pytesseract.image_to_string` to perform OCR. The extracted text is then sent as a query to the Flask API.
    *   Displays AI responses using a `typewriter` effect.
*   **Visual Feedback**: Uses `st.spinner`, `st.success`, `st.warning`, and `st.error` to provide clear feedback to the user during asynchronous operations.

### 4.5. Error Handling

The system incorporates `try-except` blocks at various levels (backend functions, Flask API, Streamlit GUI) to gracefully handle exceptions such as:

*   File not found or access errors.
*   Issues with Docker/Qdrant connectivity.
*   Errors during embedding or retrieval.
*   API request failures.

Error messages are propagated back to the user interface, providing informative feedback.

## 5. Setup and Installation

Follow these steps to set up and run the RAG AI System on your local machine.

### 5.1. Prerequisites

*   **Python 3.8+**: Ensure Python is installed on your system.
*   **pip**: Python package installer (usually comes with Python).
*   **Docker Desktop**: **Required if you plan to use Qdrant as your vector store backend.**
    *   [Download Docker Desktop](https://www.docker.com/products/docker-desktop) and follow the installation instructions for your operating system.
    *   Ensure Docker Desktop is running before starting the Qdrant backend.
*   **Tesseract OCR Engine**: **Required if you plan to use the image-to-text (OCR) feature.**
    *   **Windows**: Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).
    *   **macOS**: `brew install tesseract`
    *   **Linux (Debian/Ubuntu)**: `sudo apt-get install tesseract-ocr`
    *   Make sure `tesseract` is in your system's PATH.

### 5.2. 1. Clone the Repository

First, clone the project repository to your local machine:

```bash
git clone https://github.com/your-username/your-project-repo.git
cd your-project-repo
```
*(Replace `your-username/your-project-repo` with the actual path to your repository.)*

### 5.3. 2. Install Python Dependencies

Navigate to the project's root directory and install the required Python packages using pip. It's recommended to do this within a virtual environment.

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
# Activate the virtual environment
# On Windows: .\venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate

# Create a requirements.txt file (if you don't have one)
# Then install:
pip install -r requirements.txt
```

**`requirements.txt` content:**
```plaintext
# Core Libraries
Flask==2.3.3
requests==2.31.0
streamlit==1.28.2
pandas==2.1.3
PyPDF2==3.0.1
python-docx==1.1.0
Pillow==10.1.0 # For image processing (PIL)
pytesseract==0.3.10 # For OCR

# LangChain and related components
langchain==0.0.345
langchain-community==0.0.8 # For QdrantVectorStore, FAISS, TextLoader
langchain-openai==0.0.8 # For OpenAIEmbeddings and LLM
langchain-text-splitters==0.0.1 # For RecursiveCharacterTextSplitter (CH)
qdrant-client==1.7.0 # Qdrant Python client
faiss-cpu==1.7.4 # FAISS library for CPU
streamlit-vertical-slider==0.1.0 # For stv (if used for speed control, etc.)
```
*(Note: Pinning exact versions as shown above helps ensure reproducibility. You might adjust these based on your specific environment or if newer versions introduce breaking changes.)*

### 5.4. 3. Docker Setup for Qdrant (Mandatory for Qdrant Backend)

If you plan to use Qdrant as your vector store backend, you need to run a Qdrant container.

1.  **Ensure Docker Desktop is running** on your system.
2.  **Pull the Qdrant Docker image and run it**:
    ```bash
    docker pull qdrant/qdrant
    docker run -p 7000:6333 -p 7001:6334 qdrant/qdrant
    ```
    This command starts a Qdrant instance accessible at `http://localhost:7000` (for gRPC/HTTP API) and `http://localhost:7001` (for GRPC web). The application uses port `7000`.

### 5.5. 4. Set up OpenAI API Key

Your project uses OpenAI for embeddings and the LLM. You need to provide your OpenAI API key.

1.  **Create an `.env` file** in the root directory of your project (where your Python scripts are).
2.  **Add your OpenAI API key** to this file:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    ```
    *Replace `"your_openai_api_key_here"` with your actual API key.*
    *Make sure to import and use `os.environ.get("OPENAI_API_KEY")` or similar in your code where `API_KEY` or `emb` is initialized.*

### 5.6. 5. Run the Application

The project consists of a Flask backend and a Streamlit frontend. You need to run both simultaneously.

1.  **Start the Flask Backend**:
    Open a new terminal or command prompt, navigate to your project directory, and run your Flask app. Assuming your Flask app code is in a file named `app.py` (or similar, where `@app.route` decorators are used):
    ```bash
    python your_flask_app_file_name.py
    ```
    *(If your Flask app uses `flask run`, you might need to set `FLASK_APP` environment variable first.)*

2.  **Start the Streamlit Frontend**:
    Open another new terminal or command prompt, navigate to your project directory, and run your Streamlit app. Assuming your Streamlit app code is in a file named `streamlit_app.py` (or similar):
    ```bash
    streamlit run your_streamlit_app_file_name.py
    ```

    This will open the Streamlit application in your web browser, typically at `http://localhost:8501`.

## 6. Usage Guide

Once the Flask backend and Streamlit frontend are running, you can start interacting with the RAG AI System.

### 6.1. 1. Launching the Application

*   Ensure both your Flask backend and Qdrant Docker container (if using Qdrant) are running.
*   Open your web browser and navigate to the address provided by Streamlit (usually `http://localhost:8501`).

### 6.2. 2. Uploading a Document

1.  On the left sidebar, you will see a section titled ":rainbow[UPLOAD DOCUMENT HERE]".
2.  **Choose a vector store backend**: Select either "Qdrant" or "Faiss" from the dropdown.
3.  **Upload your document**: Click the "Browse files" button under "PDF,CVS and DOCX FILE'S ONLY" and select your `.pdf`, `.csv`, or `.docx` file.
4.  **Process the document**: Click the ":green[Upload Document]" button.
    *   A spinner will indicate "Processing Document...".
    *   The system will extract text, split it into chunks, and embed it.
    *   Progress messages like "Preparing For Page Extracting", "Contacting Servers", etc., will appear.
    *   Once complete, a ":green[UPLOAD WAS SUCCESSFUL]" message will be displayed. If any errors occur, a warning message will appear.

### 6.3. 3. Interacting with the Document

After a document is successfully uploaded and embedded, the chat interface will become active.

1.  **Type your question**: In the "Message Document AI" chat input field at the bottom, type your natural language query related to the content of your uploaded document.
2.  **Send your query**: Press Enter or click the send icon.
3.  **Receive AI response**: The AI will "type" its response based on the relevant information retrieved from your document.
    *   **Contextual Answers**: The AI will attempt to answer only from the provided document.
    *   **"I don't know"**: If the information is not found in the document, the AI is programmed to politely state that it doesn't know.
4.  **Upload an Image for OCR Query**:
    *   Click the "Attach file" (paperclip) icon in the chat input.
    *   Select a `.jpg`, `.jpeg`, or `.png` image file.
    *   The system will attempt to extract text from the image using OCR and then use that text as a query against your document. The image itself will also be displayed in the chat.

## 7. Features

*   **Adaptive Backend**: Seamlessly switch between Qdrant and FAISS for vector storage.
*   **Multi-Document Support**: Process and query various common document formats.
*   **Intelligent RAG**: Combines the power of vector search with LLMs for accurate, document-grounded answers.
*   **Intuitive UI**: Streamlit provides an easy-to-use graphical interface.
*   **Real-time Processing Feedback**: Users are informed about the progress of document processing and API calls.
*   **Robust Error Handling**: Provides informative messages for common issues (e.g., file not found, Docker not running).
*   **OCR Capability**: Enhances query input by allowing text extraction from images.

## 8. Troubleshooting

*   **"ERROR: Docker Not Running" or Qdrant Connectivity Issues**:
    *   **Solution**: Ensure Docker Desktop is installed, running, and the Qdrant container is started via `docker run -p 7000:6333 -p 7001:6334 qdrant/qdrant`. Check your firewall settings if issues persist.
*   **"Document Not Found Please Try Uploading Your Document Again" (Flask API Error)**:
    *   **Cause**: The Flask backend couldn't find the expected vector store location.
    *   **Solution**: Double-check that the document was successfully uploaded and processed, and that the returned path/collection name is correct. Ensure the `Endpoint_Files` directory exists and has proper permissions.
*   **"File is too large for Embeddings, Please upload with less text" (CSV specific warning)**:
    *   **Cause**: CSV content, when converted to a single string, might exceed limits for embedding models or lead to memory issues.
    *   **Solution**: Try splitting large CSVs into smaller files or pre-processing to extract only relevant columns/rows.
*   **"Text extracting Failed" (for Image OCR)**:
    *   **Cause**: Tesseract OCR engine might not be installed or configured correctly (not in PATH). The image quality might also be too low for accurate text extraction.
    *   **Solution**: Install Tesseract OCR (see Prerequisites) and ensure it's accessible from your system's PATH. Try with clearer images.
*   **General `requests.exceptions.ConnectionError`**:
    *   **Cause**: The Flask backend server is not running or is not accessible at `http://127.0.0.1:8000`.
    *   **Solution**: Verify that your Flask app (e.g., `python your_flask_app_file_name.py`) is running in a separate terminal and not showing any errors.
*   **`ModuleNotFoundError` or other Python package errors**:
    *   **Cause**: Not all required Python packages are installed.
    *   **Solution**: Run `pip install -r requirements.txt` again to ensure all dependencies are met. Ensure your virtual environment is active.
*   **`allow_dangerous_deserialization=True` warning**:
    *   **Note**: This warning appears for FAISS loading and indicates a potential security risk if you are loading FAISS indexes from untrusted sources. For this project, as you are generating the indexes yourself, it's generally safe. However, be aware of this for production systems with external data.

## 9. Future Enhancements

*   **Asynchronous Processing**: Implement asynchronous queues (e.g., Celery with Redis) for document processing to prevent UI blocking for very large files.
*   **Scalability**: Containerize the Flask backend and Streamlit frontend using Docker Compose for easier deployment and scaling.
*   **More LLM Options**: Allow users to choose between different LLM providers (e.g., Anthropic, Google Gemini) via configuration.
*   **Advanced Chunking Strategies**: Explore more sophisticated text splitting methods (e.g., based on semantic coherence or document structure).
*   **Source Citation**: In RAG responses, include source document names and page numbers/chunk IDs for greater transparency and verifiability.
*   **User Management**: Implement user accounts to manage document uploads and chat histories separately.
*   **Persistent Chat History**: Save chat conversations to a database.
*   **API Key Management**: Move API keys to a more secure system than environment variables for production deployments.

## 10. Contact

For any questions, suggestions, or collaboration opportunities, please feel free to reach out:

**[Your Name/Alias]**
**[Your Email Address]**
**[Your LinkedIn Profile URL (Optional)]**
**[Your GitHub Profile URL (Optional)]**

---
