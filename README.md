---
# Project Title: RAG AI System

## Table of Contents

1. [Introduction](#introduction)
2. [Project Components](#project-components)
    - [Embeddings and Vector Stores](#embeddings-and-vector-stores)
    - [API Endpoints](#api-endpoints)
    - [User Interface (UI)](#user-interface-ui)
3. [Installation and Setup](#installation-and-setup)
4. [Usage](#usage)
5. [Features](#features)
6. [Troubleshooting](#troubleshooting)
7. [Contributing](#contributing)
8. [License](#license)
9. [Contact](#contact)

## Introduction

The RAG AI System is a comprehensive tool designed to facilitate the embedding and retrieval of documents using vector stores such as Qdrant and FAISS. It supports processing and querying PDFs, CSVs, and DOCX documents through a user-friendly interface built with Streamlit.

## Project Components

### Embeddings and Vector Stores

- **OpenAIEmbeddings**: Uses OpenAI APIs for generating embeddings.
- **QdrantVectorStore**: Utilizes Qdrant for storing and retrieving vectors.
- **FAISS**: An alternative vector store for managing document embeddings.

### API Endpoints

- **/VecStore**: Handles similarity searches on vector data.
- **/textembeddings/faiss**: Manages embeddings and storage of document data.

### User Interface (UI)

- **Streamlit GUI**: Provides an interactive front-end for users to upload documents and perform queries.
- **Real-time Chat Interface**: Users can query uploaded documents and receive responses from the AI system.

## Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo.git
   ```
   
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and Install Docker**:

    [Download Docker](https://www.docker.com/products/docker-desktop/) and follow the installation instructions for your operating system.
    Ensure Docker is running for Qdrant to function properly.
    replace with the path like this 
    ``` Open The terminal in Docker : 
    docker run -d -p 7000:6333 -v "C:\Users\bruce\OneDrive\Desktop\New folder\PYTHON\Demo\Endpoint_Files\qdrant_data:/qdrant/storage" --name qdrant-rag qdrant/qdrant
    ```
4. **Run the Rag Server**
   ```
   python Rag.py
   ```

5. **Run the application**:
   ```bash
   streamlit run D_AI.py
   ```

## Usage

Upload your document through the UI, select the vector store preference, and interact with the uploaded document using the text chat interface.

## Features

- **Multi-format Support**: Handles PDF, CSV, and DOCX files.
- **Flexible Backend**: Choose between Qdrant or FAISS as vector store backends.
- **Real-time Feedback**: Provides real-time responses to user queries.

## Troubleshooting

- Ensure you have Docker running for Qdrant.
- Check that all dependencies in `requirements.txt` are installed.

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

## Contact

For further information, please contact [Bruce-Arhin Shadrach] at [mailto:brucearhin098@gmail.com].

---
