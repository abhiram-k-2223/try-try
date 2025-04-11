from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import google.generativeai as genai
import uuid
from typing import Dict, Optional

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'pdf'}
GOOGLE_API_KEY = 'Replace-with-your-API-key'  
genai.configure(api_key=GOOGLE_API_KEY)

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

    def process_pdf(self, pdf_path: str) -> int:
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
            raise HTTPException(status_code=500, detail=str(e))

    def get_study_response(self, question: str):
        if not self.conversation_chain:
            raise HTTPException(status_code=400, detail="No conversation chain available. Please process a PDF first.")

        enhanced_question = f"""
        Acting as a knowledgeable tutor, please help me understand the following question about the study material:
        {question}
        
        Please include in your response:
        1. A clear explanation of the concept
        2. Practical examples or analogies when applicable
        """

        try:
            result = self.conversation_chain({"question": enhanced_question})
            return result["answer"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

class ConversationManager:
    def __init__(self):
        self.conversations: Dict[str, StudyMaterialChat] = {}

    def create_conversation(self) -> str:
        conversation_id = str(uuid.uuid4())
        study_chat = StudyMaterialChat()
        self.conversations[conversation_id] = study_chat
        return conversation_id

    def get_conversation(self, conversation_id: str) -> Optional[StudyMaterialChat]:
        return self.conversations.get(conversation_id)

    def remove_conversation(self, conversation_id: str):
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]

conversation_manager = ConversationManager()

class ChatRequest(BaseModel):
    conversation_id: str
    question: str

def get_conversation(conversation_id: str) -> StudyMaterialChat:
    chat = conversation_manager.get_conversation(conversation_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return chat

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    conversation_id = conversation_manager.create_conversation()
    file_path = os.path.join(UPLOAD_FOLDER, f"{conversation_id}.pdf")
    
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        chat = conversation_manager.get_conversation(conversation_id)
        num_pages = chat.process_pdf(file_path)
        
        return {
            "message": "File uploaded successfully",
            "conversation_id": conversation_id,
            "pages": num_pages
        }
    except Exception as e:
        conversation_manager.remove_conversation(conversation_id)
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    chat = get_conversation(request.conversation_id)
    try:
        response = chat.get_study_response(request.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
