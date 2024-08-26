import os
import json
from werkzeug.utils import secure_filename
from transformers import AutoTokenizer, pipeline
from langchain_community.vectorstores import FAISS
from flask import request, session, render_template
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


def allowed_file(filename, ALLOWED_EXTENSIONS):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_name(app):
    directory_path1= "D:\\Practice\\RAG_POC\\Data"
    files1=os.listdir(directory_path1)
    for file in files1:
        file_path=os.path.join(directory_path1,file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        
    files = request.files['pdfFile']
    filename = secure_filename(files.filename)
    files.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return filename

def get_answer(question, UPLOAD_FOLDER, filename):
    session["context_question"]=[]
    # Load the question and split into chunks
    loader = PyPDFLoader(f"{UPLOAD_FOLDER}\\{filename}")
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=0,separators=".")
    docs = text_splitter.split_documents(pages)
    modelPath = "sentence-transformers/all-MiniLM-l6-v2"

    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    model_kwargs = {'device':'cpu'}

    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    encode_kwargs = {'normalize_embeddings': False}

    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )
    db = FAISS.from_documents(docs, embeddings)
    if request.form.get("btnTask1")=="click1":
        searchDocs = db.similarity_search(question)
        context=searchDocs[0].page_content
        context_question = [(doc.page_content, question) for doc in searchDocs]
        session["context_question"]=json.dumps(context_question)
        model_name = "Intel/dynamic_tinybert"

        # Load the tokenizer associated with the specified model
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512,max_token_length=512)

        # Define a question-answering pipeline using the model and tokenizer
        question_answerer = pipeline(
            "question-answering", 
            model=model_name, 
            tokenizer=tokenizer,
            return_tensors='pt',
            max_answer_len=512
        )
        answer=question_answerer(question=question, context=context)
        return answer, context
    
def feedback_answer_generation(context_question, j):
    model_name = "Intel/dynamic_tinybert"

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512,max_token_length=512)

    question_answerer = pipeline(
        "question-answering", 
        model=model_name, 
        tokenizer=tokenizer,
        return_tensors='pt',
        max_answer_len=512
        )
    answer=question_answerer(question=context_question[j][1], context=context_question[j][0])
    return render_template("index1.html",answer=f"answer: {answer['answer']}",context=f"context: {context_question[j][0]}",question=f"question: {context_question[j][1]}")