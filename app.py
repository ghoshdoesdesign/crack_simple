import os
from flask import Flask, request, render_template, send_from_directory, jsonify
import openai
from pyzotero import zotero
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')

# Set up OpenAI API key
openai.api_key = 'sk-68Y7HODIib0kJGpK5v9RIj2gLpasocp-VibzRNdUn-T3BlbkFJFbcqLR55JpgX6H1RSL2Osvl1DqppuKVLSUAm7IJSMA'  # Replace with your actual API key

# Set up Zotero API
zot = zotero.Zotero('10305233', 'user', 'rm5BW5WnXRfXlelUKKus1ZJP')

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
    llm = ChatOpenAI(model="gpt-4")

    # Create the ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        return_generated_question=True,
    )
    
    return chain

# Store the chains and histories in memory
chains = {}
chat_histories = {}  # To keep track of chat histories and context

# Store notes and highlights in memory (for simplicity)
# In a production app, consider using a database
notes_storage = {}
highlights_storage = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file and file.filename.lower().endswith('.pdf'):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        chains[filename] = create_chain(filepath)  # Create a chain for this PDF
        chat_histories[filename] = {'chat_history': [], 'context': ""}  # Initialize chat history and context
        notes_storage[filename] = []  # Initialize notes storage
        highlights_storage[filename] = {}  # Initialize highlights storage
        print(f"File saved to: {filepath}")
        return render_template('viewer.html', filename=filename)
    return 'Invalid file type', 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def retrieve_zotero():
    # Fetch all items from the Zotero library
    items = zot.everything(zot.items())

    # Directory to store Zotero files in
    # TODO: Add in directory
    download_dir = '' 

    # Download each attachment if it is a PDF
    for item in items:
        # Check if the item has an attachment and is a PDF
        if 'data' in item and 'contentType' in item['data'] and item['data']['contentType'] == 'application/pdf':
            file_name = item['data'].get('filename', f"{item['data']['key']}.pdf")
            file_path = os.path.join(download_dir, file_name)
            
            # Extract the item key to use with zot.file()
            item_key = item['data']['key']
            print(f"Downloading {file_name}...")
            
            # Retrieve the file
            attachment = zot.file(item_key)
            
            if attachment:
                with open(file_path, 'wb') as f:
                    f.write(attachment)
            else:
                print(f"Failed to download {file_name}.")
        else:
            print(f"Skipping item {item['data']['key']}: Not a PDF.")
    

@app.route('/chat', methods=['POST'])
def chat():
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

@app.route('/save_note', methods=['POST'])
def save_note():
    data = request.json
    filename = data.get('filename')
    note = data.get('note')

    if filename not in chains:
        return jsonify({"error": "PDF not found"}), 404

    # Store the note
    notes_storage[filename].append(note)

    return jsonify({"success": True})

@app.route('/save_highlight', methods=['POST'])
def save_highlight():
    data = request.json
    filename = data.get('filename')
    page_number = data.get('page_number')
    highlight = data.get('highlight')  # Should contain coordinates and comment

    if filename not in chains:
        return jsonify({"error": "PDF not found"}), 404

    if page_number not in highlights_storage[filename]:
        highlights_storage[filename][page_number] = []
    
    highlights_storage[filename][page_number].append(highlight)

    return jsonify({"success": True})

@app.route('/get_highlights', methods=['GET'])
def get_highlights():
    filename = request.args.get('filename')
    if not filename or filename not in chains:
        return jsonify({"error": "PDF not found"}), 404
    
    return jsonify({"highlights": highlights_storage.get(filename, {})})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
