import streamlit as st
import streamlit_vertical_slider as stv
import time,io

st.header("🤖:rainbow[RAG AI SYSTEM]",anchor="centre", divider='rainbow')

def typewriter(text: str ):
    speed=13
    tokens = text.split()
    container = st.empty()
    for index in range(len(tokens) + 1):
        curr_full_text = " ".join(tokens[:index])
        container.write(f"{curr_full_text} ✍🏾")
        time.sleep(1 / speed)
from PyPDF2 import PdfReader   
import pandas as Pd  
from docx import Document as Dc   
import requests
from PIL import Image
import pytesseract as py

session=requests.Session()
if 'message' not in st.session_state:
    st.session_state.message=[]

if "messagechat" not in st.session_state:
    st.session_state.messagechat=[]
    
for messages in st.session_state.messagechat:
    with st.chat_message(messages["Role"]):
        if messages['Role']=='image':
            st.image(messages["Content"])
        else:    
            st.markdown(messages["Content"])    

if "cvss" not in st.session_state:
    st.session_state.cvss=[]


if 'docx' not in st.session_state:
    st.session_state.docx=[]

process=st.empty()

if 'Docname' not in st.session_state:
    st.session_state.Docname=[]


with st.sidebar:
    st.divider()
    st.title(':rainbow[UPLOAD DOCUMENT HERE]',anchor='center')
    st.divider()
    text=""
    cvs=""
   
    backend_option = st.selectbox("Choose a vector store backend:",
    ("Qdrant", "Faiss"))
    st.divider()
    Document=st.file_uploader(" PDF,CVS and DOCX FILE'S ONLY",type=['pdf','cvs','docx'])
    if Document and backend_option:
        name=Document.name
        st.write(Document.type)
        dtype=Document.type.split('/').pop()
        if st.button(":green[Upload Document]"):
            st.session_state.Docname.clear()
            st.session_state.message.clear()
            st.session_state.cvss.clear()
            st.session_state.docx.clear()
            if dtype=="pdf":
                
                
                    
                
                    try:
                        with st.spinner("Processing Document..... "):
                            
                            #Reading Document
                            Data=PdfReader(Document)
                            text=" "
                            
                            
                            process.write("Preparing For Page Extracting ")
                            
                            #Processing Document
                            for pages in Data.pages:
                                text+=pages.extract_text()
                            process.write("Done Extracting Pages....")
                            time.sleep(2)
                            process.write("Contacting Servers ")
                            for i in range(6):
                                process.write("Contacting Servers "+"."*i)
                                time.sleep(1)
                                
                            #Making API Request
                            st.session_state.message.append({"PDF":text})    
                            endpoint=session.post('http://127.0.0.1:8000/textembeddings/faiss', json={'Name':name.split('.')[0],'Document':st.session_state.message[0]['PDF'],'Vstore':backend_option})
                            if endpoint.status_code==200:
                                message=endpoint.text
                                st.session_state.Docname.append({'File':message})
                                
                                st.success(f":green[UPLOAD WAS SUCESSFUL]")
                            elif endpoint.status_code==500:
                                st.warning(endpoint.text,icon="🚨")
                            process.empty()            
                    except Exception as ex:
                        st.warning(ex,icon="🚨")
            elif dtype=="csv":
            
                    try:
                        with st.spinner("Processing Document....."):
                            
                            process.write("Preparing For Page Extracting ")
                            
                            #Reading Document
                            Data=Pd.read_csv(Document)
                            cvs=" "
                            
                            
                            #Processing Document
                            for Data in Data.to_string():
                                cvs+=Data
                            st.session_state.cvss.append({"CVS":cvs}) 
                            
                            process.write("Done Extracting Pages")
                            time.sleep(2)
                            for i in range(6):
                                process.write("Contacting Servers"+".."*i)
                                time.sleep(1)
                                
                        
                            #Making API Request                        
                            #Making API Request
                            endpoint=session.post('http://127.0.0.1:8000/textembeddings/faiss', json={'Name':name.split('.')[0],'Document':st.session_state.cvss[0]['CVS'],'Vstore':backend_option})
                            if endpoint.status_code==200:
                                message=endpoint.text
                                
                                
                                st.session_state.Docname.append({'File':message})
                                st.success(f":green[UPLOAD WAS SUCESSFUL]")
                                process.success(f":green[UPLOAD WAS SUCESSFUL]")
                            elif endpoint.status_code==500:
                                st.warning("File is too large for Embeddings, Please upload with less text",icon="🚨")
                                    
                            process.empty()  
                    except Exception as ex:
                        st.warning(ex,icon="🚨")
            elif dtype=='vnd.openxmlformats-officedocument.wordprocessingml.document':
                    try:
                        with st.spinner("Processing Document....."):
                            
                            #Reading Document
                            texts=" "
                            doc=Dc(Document)
                            
                            #Processing Document
                            for item in doc.paragraphs:
                                texts+=(f"\n{item.text}")
                            for _ in doc.tables:
                                for Row in _.rows:
                                    for cell in Row.cells:
                                        texts+=(f"\n{cell.text}")  
                                        
                            
                            st.session_state.docx.append({'DOCX':texts})
                            
                            
                            #Making API Request
                            endpoint=session.post('http://127.0.0.1:8000/textembeddings/faiss', json={'Name':name.split('.')[0],'Document':st.session_state.docx[0]['DOCX'],'Vstore':backend_option})
                            if endpoint.status_code==200:
                                message=endpoint.text
                                
                                
                                
                                st.session_state.Docname.append({'File':message})
                                st.success(f":green[UPLOAD WAS SUCESSFUL]")              
                    except Exception as ex:
                        st.warning(ex,icon="🚨")                
            else:
                st.warning(":red[DOCUMENT MUST BE A PDF OR CSV FILE]",icon="🚨")    



if st.session_state.Docname:
    chat=st.chat_input("Message Document AI",accept_file=True,file_type=["jpg", "jpeg", "png"])
    if chat and chat.text:
        
        with st.spinner("Going Through Your Document"):
            try:
                text=chat.text
                Chain=session.post('http://127.0.0.1:8000/VecStore',json={"Vector":st.session_state.Docname[0]['File'],'Query':text})
                if Chain.status_code==200:
                    # Add User input to chat history
                    st.session_state.messagechat.append({"Role": "Human", "Content": text}) 
                    messag=Chain.text
                    respond=f"ChatBot: {messag}"
                    with st.chat_message("AI"):
                        typewriter(text=respond)
                    # Add assistant response to chat history    
                    st.session_state.messagechat.append({"Role": "AI", "Content": messag})          
                else: 
                            message=Chain.text   
                            st.write(f":red[{message}]")     
            except Exception as ex:
                st.warning(ex,icon="🚨")
    elif chat and chat['files']:
        img=io.BytesIO(chat['files'][0].read())
        img=Image.open(img)
        pro=py.image_to_string(img.convert("L"))
        if pro:
            st.image(img)
            st.session_state.messagechat.append({"Role": "image", "Content": img}) 
            
            with st.spinner("Going Through Your Document"):
                try:
                    
                    Chain=session.post('http://127.0.0.1:8000/VecStore',json={"Vector":st.session_state.Docname[0]['File'],'Query':pro})
                    if Chain.status_code==200:
                        # Add User input to chat history
                        messag=Chain.text
                        respond=f"ChatBot: {messag}"
                        with st.chat_message("AI"):
                            typewriter(text=respond)
                        # Add assistant response to chat history    
                        st.session_state.messagechat.append({"Role": "AI", "Content": messag})          
                    else: 
                                message=Chain.text   
                                st.write(f":red[{message}]")     
                except Exception as ex:
                    st.warning(ex,icon="🚨")
        else:
            st.warning("Text extracting Failed ",icon="🚨")
                        
                
        
        
                    
        