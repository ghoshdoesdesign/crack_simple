from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, flash
import os

import openai
from openai import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

app = Flask(__name__)

# Configure the upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Store the chains and histories in memory
chains = {}
chat_histories = {}  # To keep track of chat histories and context

# Store notes and highlights in memory (for simplicity)
# In a production app, consider using a database
notes_storage = {}
highlights_storage = {}


# Function to create LangChain pipeline
def create_chain(pdf_path):
    # Load PDF and extract text
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Create the vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = FAISS.from_documents(docs, embedding=embeddings)

    # Set up the LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # Create the ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        return_generated_question=True,
    )
    
    return chain

# Route to handle file upload and parsing
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file and file.filename.lower().endswith('.pdf') and file.filename != ".DS_Store":
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        chains[filename] = create_chain(filepath)  # Create a chain for this PDF
        chat_histories[filename] = {'chat_history': [], 'context': ""}  # Initialize chat history and context
        notes_storage[filename] = []  # Initialize notes storage
        highlights_storage[filename] = {}  # Initialize highlights storage
        print(f"File saved to: {filepath}")
        print(chains.keys())
        return redirect(url_for('index'))
    return 'Invalid file type', 400










# Route for the landing page
@app.route('/')
def landing_page():
    return render_template('landing.html')

# Home route to display the list of PDFs and upload option
@app.route('/pdf-viewer', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle PDF upload
        file = request.files['file']
        if file and file.filename.endswith('.pdf'):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            return redirect(url_for('index'))
    
    # Get the list of uploaded PDFs
    pdf_list = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('index.html', pdfs=pdf_list)

# Route to serve the uploaded PDFs
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



def chat_dep1():
    data = request.json
    filename = data.get('filename')
    query = data.get('query')

    if filename not in chains:
        return jsonify({"error": "PDF not found"}), 404

    chain = chains[filename]
    history = chat_histories.get(filename, {'chat_history': [], 'context': ""})
    chat_history = history['chat_history']

    # Use the chat history to maintain context
    inputs = {
        'question': query,
        'chat_history': chat_history
    }
    
    response = chain(inputs)

    # Extract the answer and the generated question (which serves as context)
    answer = response['answer']
    generated_question = response.get('generated_question', "")

    # Update chat history with the new question, answer, and generated question
    chat_history.append((query, answer))
    chat_histories[filename] = {
        'chat_history': chat_history, 
        'context': generated_question
    }

    return jsonify({
        "response": answer,
        "context": generated_question
    })


def chat_dep2():
    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Write a haiku about recursion in programming."
            }
        ]
    )
    print(completion.choices[0].message.content)
    return jsonify({
        "response": completion.choices[0].message.content
    })


@app.route('/chat', methods=['POST'])
def chat_dep():
    data = request.json
    query = data.get('query')

    # Log the received query
    print(f"Received query: {query}")

    # Hardcoded response
    response = "This is a hardcoded response from Ptolemy. This is where we will see the long AI-generated response. Honestly, I'm just dragging this out to see what the response looks like with multiple lines. I am quite sleepy."

    return jsonify({
        "response": response
    })


if __name__ == '__main__':
    app.run(debug=True)
