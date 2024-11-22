from flask import Flask, request, jsonify, render_template, session
from transformers import pipeline, BertTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import os
import fitz
import logging
import tempfile
import numpy as np

app = Flask(__name__)
app.secret_key = 'abhiram'

# Enhanced logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Load models
try:
    qa_pipeline = pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad')
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logging.info("Models loaded successfully")
except Exception as e:
    logging.error(f"Error loading models: {str(e)}")
    raise

# Store user sessions in-memory
user_sessions = {}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload_document():
    if 'document' not in request.files:
        return jsonify({'error': 'No document part in the request'}), 400
    
    file = request.files['document']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are supported'}), 400

    try:
        # Create a temporary file to handle the PDF properly
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            file.save(tmp_file.name)
            logging.debug(f"Temporary file created at: {tmp_file.name}")
            
            # Extract text from the PDF
            content = extract_text_from_pdf(tmp_file.name)
            
            # Clean up the temporary file
            os.unlink(tmp_file.name)

        if not content.strip():
            return jsonify({'error': 'No text could be extracted from the PDF'}), 400

        # Preprocess and index the document
        chunks = preprocess_document(content)
        if not chunks:
            return jsonify({'error': 'Document preprocessing failed'}), 500

        # Create the index and embeddings
        try:
            index = create_index(chunks)
            logging.debug("Index created successfully")
        except Exception as e:
            logging.error(f"Error creating index: {str(e)}")
            return jsonify({'error': 'Failed to create search index'}), 500

        # Store document data in the session
        session_id = session.get('session_id')
        if not session_id:
            session_id = os.urandom(24).hex()
            session['session_id'] = session_id

        user_sessions[session_id] = {
            'content': content,
            'chunks': chunks,
            'index': index
        }

        return jsonify({'message': 'Document uploaded successfully!'})

    except Exception as e:
        logging.error(f"Error processing document: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error processing document: {str(e)}'}), 500

def extract_text_from_pdf(pdf_path):
    try:
        logging.debug(f"Attempting to open PDF at: {pdf_path}")
        # Open the PDF file with PyMuPDF (fitz)
        doc = fitz.open(pdf_path)
        text = []

        # Iterate through each page in the PDF and extract text
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            page_text = page.get_text("text")
            text.append(page_text)
            logging.debug(f"Extracted {len(page_text)} characters from page {page_num + 1}")

        doc.close()
        # Clean the text by removing extra whitespaces and newlines
        cleaned_text = ' '.join(' '.join(text).split())
        logging.debug(f"Total extracted text length: {len(cleaned_text)}")
        return cleaned_text
    
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {str(e)}", exc_info=True)
        raise Exception(f"Failed to extract text from PDF: {str(e)}")

# def preprocess_document(document):
#     try:
#         tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         sentences = document.split('.')
#         chunks = []
#         current_chunk = []
        
#         for sentence in sentences:
#             if not sentence.strip():
#                 continue
                
#             tokens = tokenizer.tokenize(sentence)
#             if len(current_chunk) + len(tokens) <= 512:
#                 current_chunk.extend(tokens)
#             else:
#                 chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
#                 current_chunk = tokens
                
#         if current_chunk:
#             chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
            
#         logging.debug(f"Created {len(chunks)} chunks from document")
#         return chunks
    
#     except Exception as e:
#         logging.error(f"Error in preprocessing document: {str(e)}", exc_info=True)
#         raise Exception(f"Failed to preprocess document: {str(e)}")

def preprocess_document(document):
    """
    Preprocess document by splitting it into manageable chunks with overlap.
    """
    try:
        if not document or not isinstance(document, str):
            raise ValueError("Invalid document input")
            
        logging.debug(f"Processing document of length: {len(document)}")
        
        # Split into sentences first (more reliable than paragraphs)
        sentences = [s.strip() for s in document.split('.') if s.strip()]
        chunks = []
        current_chunk = []
        current_length = 0
        
        logging.debug(f"Number of sentences: {len(sentences)}")
        
        # Process sentences with overlap
        for sentence in sentences:
            words = sentence.split()
            word_count = len(words)
            
            # If adding this sentence would exceed chunk size
            if current_length + word_count > 300:  # Reduced chunk size
                if current_chunk:
                    chunk_text = ' '.join(current_chunk) + '.'
                    chunks.append(chunk_text)
                    # Keep last sentence for overlap
                    current_chunk = [current_chunk[-1], sentence] if current_chunk else [sentence]
                    current_length = len(current_chunk[-1].split())
                else:
                    # Handle case where single sentence exceeds length
                    chunks.append(sentence + '.')
                    current_chunk = []
                    current_length = 0
            else:
                current_chunk.append(sentence)
                current_length += word_count
        
        # Add the last chunk if it exists
        if current_chunk:
            chunk_text = ' '.join(current_chunk) + '.'
            chunks.append(chunk_text)
        
        # Validate chunks
        chunks = [chunk for chunk in chunks if len(chunk.split()) >= 10]  # Minimum chunk size
        
        if not chunks:
            raise ValueError("No valid chunks were created from the document")
        
        logging.debug(f"Created {len(chunks)} chunks")
        logging.debug(f"Sample chunk: {chunks[0][:100]}")
        logging.debug(f"Average chunk size: {sum(len(chunk.split()) for chunk in chunks) / len(chunks)} words")
        
        return chunks
    
    except Exception as e:
        logging.error(f"Error in preprocessing document: {str(e)}", exc_info=True)
        raise

def get_relevant_chunks(query, index, chunks, k=5, max_context_length=4000):
    """
    Get more relevant chunks for comprehensive answers
    """
    try:
        if not chunks:
            raise ValueError("No document chunks available")
            
        logging.debug(f"Processing query: {query}")
        logging.debug(f"Available chunks: {len(chunks)}")
        
        # Get query embedding
        query_embedding = embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Increase number of retrieved chunks for more comprehensive answers
        k = min(k, len(chunks))
        distances, indices = index.search(query_embedding, k)
        
        # Sort chunks by relevance
        chunk_scores = list(zip(indices[0], distances[0]))
        chunk_scores.sort(key=lambda x: x[1])  # Sort by distance (lower is better)
        
        relevant_chunks = []
        total_length = 0
        seen_content = set()
        
        for idx, score in chunk_scores:
            if idx >= len(chunks):
                continue
                
            chunk = chunks[idx].strip()
            # Avoid duplicate content
            if chunk in seen_content:
                continue
                
            chunk_length = len(chunk)
            if total_length + chunk_length <= max_context_length:
                relevant_chunks.append({
                    'text': chunk,
                    'score': float(score)
                })
                seen_content.add(chunk)
                total_length += chunk_length
        
        if not relevant_chunks:
            raise ValueError("Could not find relevant chunks")
        
        # Combine chunks with weighting by relevance score
        combined_context = ' '.join(chunk['text'] for chunk in relevant_chunks)
        
        logging.debug(f"Final context length: {len(combined_context)}")
        return combined_context

    except Exception as e:
        logging.error(f"Error retrieving relevant chunks: {str(e)}")
        raise

@app.route('/query', methods=['POST'])
def query_document():
    try:
        session_id = session.get('session_id')
        if not session_id or session_id not in user_sessions:
            return jsonify({'error': 'No document uploaded'}), 400

        query = request.json.get('query', '').strip()
        if not query:
            return jsonify({'error': 'No query provided'}), 400

        user_data = user_sessions[session_id]
        chunks = user_data.get('chunks', [])
        index = user_data.get('index')

        if not chunks or not index:
            return jsonify({'error': 'Document data is missing'}), 400

        # Get expanded context
        relevant_context = get_relevant_chunks(query, index, chunks, k=5)
        
        # Break down complex queries into sub-questions
        sub_questions = [
            query,  # Original question
            f"What are the main points about {query}?",  # Get main points
            f"Can you provide examples related to {query}?",  # Get examples
            f"What additional details are important about {query}?"  # Get additional details
        ]
        
        # Get answers for each sub-question
        answers = []
        for sub_q in sub_questions:
            try:
                answer = qa_pipeline(
                    question=sub_q,
                    context=relevant_context
                )
                if answer and answer['answer'].strip():
                    answers.append(answer['answer'].strip())
            except Exception as e:
                logging.warning(f"Error processing sub-question '{sub_q}': {str(e)}")
                continue
        
        # Combine answers into a coherent response
        final_answer = " ".join(answers)
        
        # Format the answer with bullet points if it's long enough
        if len(final_answer.split()) > 20:
            sentences = [s.strip() for s in final_answer.split('.') if s.strip()]
            formatted_answer = "• " + "\n• ".join(sentences)
        else:
            formatted_answer = final_answer
            
        response = {
            'answer': formatted_answer,
            'confidence': float(qa_pipeline(question=query, context=relevant_context)['score']),
            'context': relevant_context[:1000]  # Include more context
        }
        
        return jsonify(response)

    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        return jsonify({'error': str(e)}), 500

def create_index(chunks):
    """
    Create FAISS index with improved error handling.
    """
    try:
        if not chunks:
            raise ValueError("No chunks provided for indexing")
            
        logging.debug(f"Creating embeddings for {len(chunks)} chunks")
        
        # Generate embeddings
        embeddings = embedding_model.encode(chunks)
        embeddings = np.array(embeddings).astype('float32')
        
        if embeddings.size == 0:
            raise ValueError("Failed to generate embeddings")
            
        # Create and populate index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        logging.debug(f"Created index with {index.ntotal} vectors")
        
        return index
        
    except Exception as e:
        logging.error(f"Error creating index: {str(e)}")
        raise

@app.route('/clear', methods=['POST'])
def clear_session():
    session_id = session.get('session_id')
    if session_id in user_sessions:
        del user_sessions[session_id]
    session.clear()
    return jsonify({'message': 'Session cleared'})

if __name__ == '__main__':
    app.run(debug=True)