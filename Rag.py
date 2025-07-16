from flask import Flask, request,url_for, jsonify,render_template,redirect,send_from_directory,send_file,Blueprint,session
from flask_cors import CORS
from bruceIQ import Database
import os,logging
from langchain_openai import OpenAIEmbeddings,OpenAI,ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_qdrant import Qdrant,QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.text_splitter import CharacterTextSplitter as CH
from qdrant_client.http.models import Distance,VectorParams
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

API_KEY=os.environ.get('OPENAI_API_KEY','')
if not API_KEY:
    raise ValueError("NO OPENAI_API_KEY FOUND")
emb=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
Embeddings=OpenAIEmbeddings(api_key=API_KEY)
llm=ChatOpenAI(model="gpt-4o",temperature=0.9,api_key=API_KEY) 
import time as t
def Qdrant_Retrival(Location, Query):
    """
    Load the locally saved vector store from its file location (Qdrant or FAISS)
    and start a RAG operation.
    """
    
    try:
        print("\n🔍Checking file and loading vector store\n")
        print("📂 Location:", Location)

        # Auto-detect if it's a Qdrant collection by folder naming convention
        if "qdrant" in Location.lower() :
            print("🔍 Detected Qdrant Vector Store")
            

            client = QdrantClient(url="http://localhost:7000")
            newDB = QdrantVectorStore(
                client=client,
                collection_name=Location,
                embedding=emb
            )
        else:
            print("🔍 Detected FAISS Vector Store")
            newDB = FAISS.load_local(Location, embeddings=emb, allow_dangerous_deserialization=True)

        print("✅ Vector store loaded successfully\n")
        t.sleep(1)

        # Set up Retriever and Prompt Chain
        QA = newDB.as_retriever()
        templatew = """You are a conversational AI that chats with users like hello hi  and also answers questions only from the provided document 
                        if the question is not in the document say i dont know is not in the document provided use a polite way .

                    Context:
                    {context}

                    Question: {question}
                    """
        promptt = ChatPromptTemplate.from_template(templatew)

        RAG = (
            {"context": QA, "question": RunnablePassthrough()} |
            promptt |
            llm |
            StrOutputParser()
        )

        print("🔍 Performing similarity search with RAG pipeline...")
        t.sleep(1)

        query = RAG.invoke(Query)

        # Close Qdrant client if used
        if "client" in locals():
            client.close()

        del newDB
        return query

    except Exception as ex:
        return f"ERROR: {str(ex)}"



def Qdrant_Encrypy(namee):
    """
    Load a .txt file, split and embed it, then save to Qdrant.
    If the document has already been embedded, return its collection name.
    """
    
    # Initialize Qdrant client (local or cloud)
    
    name = f'{namee}.txt'
    collection_name = f"{namee}_qdrant"
    path = f"./Endpoint_Files/{name}"
    
    if not os.path.exists(path):
        return "Document does not exist try uploading a new one."
        
    try:
        client = QdrantClient(
        url="http://localhost:7000",  #  Qdrant Docker Cloud URL 
    )
        print("🔍 Checking if document is already embedded in Qdrant...\n")
        t.sleep(1)
        # Check if collection already exists
        if not client.collection_exists(collection_name):
            print(f"📦 Creating Qdrant collection: `{collection_name}` ...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
            print(f"✅ Collection `{collection_name}` created successfully! 🚀\n")
        else:
            print(f"📁 Collection `{collection_name}` already exists. Skipping creation ✅\n")
            return f"{collection_name}"
    except Exception as ex:
        print(f"Error {str(ex)}")
        return f"Error: Docker Not Running"
        
    try:    
        # Proceed to embed
        print(f"📄 Document `{name}` will now be split and embedded...\n")
        loader = TextLoader(file_path=path, encoding="utf-8")
        text_splitter = CH(chunk_size=1000, chunk_overlap=100)
        
        print(f"📦 Loading document from {path}...\n")
        t.sleep(2)
        documents = loader.load()
        
        print("✂️ Splitting document into chunks...\n")
        t.sleep(2)
        docs = text_splitter.split_documents(documents)

        print("🧠 Embedding chunks and saving to Qdrant...\n")
        t.sleep(2)
    
        
        Vectorstore=QdrantVectorStore(
        
            embedding=emb,
            client=client,
            collection_name=collection_name
        )
        
        Vectorstore.add_documents(  documents=docs)
        client.close()
        print(f"✅ Document `{name}` embedded and stored as collection `{collection_name}`.\n")
        return f"{collection_name}"
    except Exception as ex:
        client.delete_collection(collection_name)
        return f"Error: {str(ex)}"
        





def Encrypy(namee):
    "Load a txt file split and save to faiss vectorsore and return the location of the saved local faiss vectorstore folder path"
    name=f'{namee}.txt'
    path=f"./Endpoint/{name}"
    print("🔍 Checking if document is already embedded in faiss...\n")
    t.sleep(3)
    paths='C:/Users/bruce/OneDrive/Desktop/New folder/PYTHON/Demo/Endpoint_Files/faiss/'
    if not os.path.exists(paths):
        os.makedirs(paths)
    if os.path.exists(f'{paths}{namee}'):
        print(f"📁 Faiss `{namee}` already exists. Skipping creation ✅\n")
        return (f'{paths}{namee}')
    
    else:
        print(f"📄 Document `{name}` will now be split and embedded...\n")
        Directory=TextLoader(file_path=f'./Endpoint_Files/{name}',encoding="utf-8")
        text_splitter = CH(chunk_size=1000, chunk_overlap=100)
        print(f"📦 Loading document from {path}...\n")
        t.sleep(2)
        Document=Directory.load()
        t.sleep(3)
        print("✂️ Splitting document into chunks...\n")
        t.sleep(2)
        docs = text_splitter.split_documents(Document)
        t.sleep(3)
        print(" * File Splitted Sucessfully ✔️\n") 
        t.sleep(2)
        print("🧠 Embedding chunks and saving to Faiss...\n")
        DB=FAISS.from_documents(docs,emb)
      
        t.sleep(3)
        DB.save_local(f'{paths}{namee}')
        t.sleep(2)
        print(f"✅ Document `{name}` embedded and stored at `{paths}{namee}`.\n")
      
        return(f'{paths}{namee}')
 




app = Flask(__name__)
CORS(app)













@app.route("/VecStore",methods=["POST",'GET'])
def Similarity_Search():
    Data=request.json
    Loc=Data.get("Vector")
    Query=Data.get("Query")
    print(Loc)
    
    if 'qdrant' in Loc:
        out=Qdrant_Retrival(Location=Loc,Query=Query)
        if out.upper().startswith('ERROR:'):
            return out,500
        else:
            print(out)
            return out   
    elif os.path.exists(Loc):
        out=Qdrant_Retrival(Location=Loc,Query=Query)
        if out.upper().startswith('ERROR:'):
            print(out)
            return out,500
       
        else:
            return out          
    else:
        print("FileNotFoundError")
        return "Document Not Found Please Try Uploading Your Document Again ",500   
   


@app.route("/textembeddings/faiss",methods=["POST","GET"])
def Splitting(): 
  Data=request.get_json()
  Doc=Data["Document"]
  Name=Data["Name"]
  Vstore=Data['Vstore']

      
      
  
  if os.path.exists(f"Endpoint_Files/{Name}.txt"):
      if Vstore=="Qdrant":
        out=Qdrant_Encrypy(Name)
        if "Error" in out:
            del Data
            return out,500
      else:    
        out=Encrypy(Name)
        if "Error" in out:
            del Data
            return out,500
      del Data
      return out
      
  else:
    with open(f'Endpoint_Files/{Name}.txt','w+',encoding='utf-8') as file:
        print("\n ******** Writing File to Storage ********\n")
        for _ in Doc:
            
            file.write(_)   
   
    t.sleep(1)
    print(f" * File path: Endpoint_File/{Name}\n")
    print( " * File has been saved With A Successfull feedback\n") 
    if Vstore=="Qdrant":
        out=Qdrant_Encrypy(Name)
        if "Error" in out:
            del Data
            return out,500
    else:
        out=Encrypy(namee=Name)
        if "Error" in out:
            del Data
            return out,500
    del Data  
    return out



if __name__ == "__main__":
    

    
    print(" \n * Starting Server\n")
    t.sleep(0.5)
    print(" * Server Is Up And Running\n")
    t.sleep(1)
    logger = logging.getLogger("Flask")
    logger.setLevel(logging.INFO)
    # app.run('0.0.0.0',port=8000)
    app.run(host="0.0.0.0", port=8000,debug=True)  