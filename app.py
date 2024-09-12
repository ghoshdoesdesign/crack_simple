from flask import Flask, jsonify, render_template, request, redirect, url_for, send_file
from dotenv import load_dotenv
import os
from flask_cors import CORS
from google.cloud import storage
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import datetime
import io
import tempfile

load_dotenv()
app = Flask(__name__)
CORS(app)

app.config['DEBUG'] = os.getenv('FLASK_DEBUG')

BUCKET_NAME = 'pdfbucketresearch'
client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)

# Store the chains and histories in memory
chains = {}
chat_histories = {}  # To keep track of chat histories and context

# Store notes and highlights in memory (for simplicity)
# In a production app, consider using a database
notes_storage = {}
highlights_storage = {}

# llm = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

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
            bucket_name = 'pdfbucketresearch'  # Replace with your bucket name
            destination_blob_name = file.filename  # You can modify this to change the destination name in the bucket
            public_url = upload_to_bucket(bucket_name, file, destination_blob_name)
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            return redirect(url_for('index'))
    print(list_bucket_items('pdfbucketresearch'))
    return render_template('index.html', pdfs=list_bucket_items('pdfbucketresearch'))

# Upload file to Google Cloud Storage
def upload_to_bucket(bucket_name, source_file, destination_blob_name):
    client = init_storage_client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_file(source_file)
    return blob.public_url

# Initialize Google Cloud Storage client
def init_storage_client():
    return storage.Client()

# List all items (blobs) in the bucket
def list_bucket_items(bucket_name):
    client = init_storage_client()
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs()
    
    # Store all blob names in a list
    items = [blob.name for blob in blobs]
    return items

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    print(file.filename)
    if file and file.filename.lower().endswith('.pdf'):
        filename = file.filename
        bucket_name = 'pdfbucketresearch'  # Replace with your bucket name
        public_url = upload_to_bucket(bucket_name, file, filename)
        # print(f"File saved to: {filepath}")
        
        chains[filename] = create_chain(filename)  # Create a chain for this PDF
        chat_histories[filename] = {'chat_history': [], 'context': ""}  # Initialize chat history and context
        notes_storage[filename] = []  # Initialize notes storage
        highlights_storage[filename] = {}  # Initialize highlights storage

        return redirect(url_for('index'))
    return 'Invalid file type', 400

# Route to serve the uploaded PDFs from Google Cloud Storage
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    client = init_storage_client()
    bucket = client.get_bucket('pdfbucketresearch')  # Replace with your bucket name
    blob = bucket.blob(filename)
    
    content = blob.download_as_bytes()
    return send_file(
        io.BytesIO(content),
        mimetype='application/pdf',
        as_attachment=False,
        download_name=filename
    )


    
# Function to create LangChain pipeline
def create_chain(filename):
	# Initialize Google Cloud Storage client
	client = init_storage_client()
	bucket = client.get_bucket('pdfbucketresearch')  # Replace with your bucket name
	blob = bucket.blob(filename)
	
	# Download the PDF content to a temporary file
	with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
		blob.download_to_file(temp_file)
		temp_file_path = temp_file.name
	
	# Use the temporary file path with PyPDFLoader
	loader = PyPDFLoader(temp_file_path)
	documents = loader.load()
	
	# Clean up the temporary file
	os.unlink(temp_file_path)

	# Split the text into smaller chunks
	text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
	docs = text_splitter.split_documents(documents)

	# Create the vector store
	embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
	vector_store = FAISS.from_documents(docs, embedding=embeddings)

	# Set up the LLM

	# Create the ConversationalRetrievalChain
	chain = ConversationalRetrievalChain.from_llm(
		llm=llm,
		retriever=vector_store.as_retriever(),
		return_source_documents=True,
		return_generated_question=True,
	)

	return chain


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    filename = data.get('pdf_name')
    query = data.get('query')
    print(filename)
    print(chains.keys())

    if filename not in chains.keys():
        return jsonify({"error": "PDF not found"}), 404

    chain = chains[filename]
    history = chat_histories.get(filename, {'chat_history': [], 'context': ""})
    chat_history = history['chat_history']

    response = chain.invoke({
        'question': query,
        'chat_history': chat_history
    })

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
        "response": answer
    })

if __name__ == "__main__":
    app.run(debug=True)