import os
from flask import Flask, request, render_template, send_from_directory, jsonify
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.question.generation import QuestionGeneratorChain

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')

# Set up OpenAI API key
openai.api_key = 'sk-68Y7HODIib0kJGpK5v9RIj2gLpasocp-VibzRNdUn-T3BlbkFJFbcqLR55JpgX6H1RSL2Osvl1DqppuKVLSUAm7IJSMA'

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

    # Load the question generation chain
    question_generator = QuestionGeneratorChain.from_llm(llm)

    # Load the QA chain
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    
    # Create the ConversationalRetrievalChain
    chain = ConversationalRetrievalChain(
        retriever=vector_store.as_retriever(),
        combine_docs_chain=qa_chain,
        question_generator=question_generator
    )
    
    return chain

# Store the chains in memory for each PDF
chains = {}

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
        print(f"File saved to: {filepath}")
        return render_template('viewer.html', filename=filename)
    return 'Invalid file type', 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    filename = data.get('filename')
    query = data.get('query')

    if filename not in chains:
        return jsonify({"error": "PDF not found"}), 404

    chain = chains[filename]
    response = chain.run(input=query)
    
    return jsonify({"response": response})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
