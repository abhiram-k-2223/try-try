from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import google.generativeai as genai
import uuid

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'pdf'}


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Replace with your API key
genai.configure(api_key=GOOGLE_API_KEY)

class ConversationManager:
    def __init__(self):
        self.conversations = {}

    def create_conversation(self):
        conversation_id = str(uuid.uuid4())
        study_chat = StudyMaterialChat()
        self.conversations[conversation_id] = study_chat
        return conversation_id

    def get_conversation(self, conversation_id):
        return self.conversations.get(conversation_id)

    def remove_conversation(self, conversation_id):
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]

class StudyMaterialChat:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7,
            convert_system_message_to_human=True
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.memory = ConversationBufferMemory(
            memory_key='chat_history',
            output_key='answer', 
            return_messages=True
        )
        self.conversation_chain = None

    def process_pdf(self, pdf_path):
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()

            chunks = self.text_splitter.split_text(text)
            vectorstore = FAISS.from_texts(texts=chunks, embedding=self.embeddings)

            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                memory=self.memory,
                return_source_documents=True,
                verbose=True
            )

            return len(reader.pages)

        except Exception as e:
            print(f"Error in process_pdf: {str(e)}")
            raise

    def get_study_response(self, question):
        if not self.conversation_chain:
            raise ValueError("No conversation chain available. Please process a PDF first.")

        enhanced_question = f"""
        Acting as a knowledgeable tutor, please help me understand the following question about the study material:
        {question}
        
        Please include in your response:
        1. A clear explanation of the concept
        2. Practical examples or analogies when applicable
        3. Key points to remember
        4. Any relevant formulas or definitions
        5. Study tips related to this topic
        """

        try:
            response = self.conversation_chain({"question": enhanced_question})
            return {
                "answer": response['answer'],
                "has_source": bool(response.get('source_documents', [])),
                "success": True
            }
        except Exception as e:
            print(f"Error in get_study_response: {str(e)}")
            return {
                "answer": f"Error processing your question: {str(e)}",
                "has_source": False,
                "success": False
            }

conversation_manager = ConversationManager()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            conversation_id = conversation_manager.create_conversation()
            study_chat = conversation_manager.get_conversation(conversation_id)

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            num_pages = study_chat.process_pdf(filepath)

            session['conversation_id'] = conversation_id

            return jsonify({
                'success': True,
                'message': f'Successfully processed PDF ({num_pages} pages)',
                'num_pages': num_pages
            })

        except Exception as e:
            if 'conversation_id' in locals():
                conversation_manager.remove_conversation(conversation_id)
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        question = data.get('question', '')

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        conversation_id = session.get('conversation_id')
        if not conversation_id:
            return jsonify({'error': 'Please upload a PDF first'}), 400

        study_chat = conversation_manager.get_conversation(conversation_id)
        if not study_chat:
            return jsonify({'error': 'Conversation not found'}), 404

        response = study_chat.get_study_response(question)
        return jsonify(response)

    except Exception as e:
        print(f"Error in chat route: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)