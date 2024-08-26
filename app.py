from flask import Flask, request, render_template

import json
from werkzeug.utils import secure_filename
from transformers import AutoTokenizer, pipeline
from langchain_community.vectorstores import FAISS
from flask import request, session, render_template
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.utils import get_file_name, get_answer, feedback_answer_generation, allowed_file

app = Flask(__name__, template_folder='Templates')
j=0

app.secret_key = 'SECRET_KEY_001_###'

UPLOAD_FOLDER = 'D:\\Practice\\RAG_POC\\Data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['pdf'])


@app.route("/",methods=["GET","POST"])
def login():
    return render_template("index2.html")

@app.route('/upload', methods=["GET",'POST'])
def upload():
    filename = get_file_name(app)
    if allowed_file(filename, ALLOWED_EXTENSIONS):
        question = request.form.get("textInput")
        answer, context = get_answer(question, UPLOAD_FOLDER, filename)
        return render_template("index1.html",answer=f"answer: {answer['answer']}",context=f"context: {context}",question=f"question: {question}")
    else:
        return render_template("index1.html",answer="Ender a PDF file only")   

@app.route('/upload/feedback', methods=["GET",'POST'])
def feedback():
    context_question = json.loads(session['context_question'])
    global j 
    if request.form.get("btnTask3")=="click3":
        
        j+=1
        if j <len(context_question):
            return feedback_answer_generation(context_question, j)
        else:
            j=0
            return render_template("index1.html",answer="Feedback given limit exceed, please restart")

    elif request.form.get("btnTask2")=="click2":
        return render_template("index1.html",answer="Thank you for your positive feedback")   

if __name__ == '__main__':
    app.run(debug=True)