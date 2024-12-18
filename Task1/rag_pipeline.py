import os
import faiss
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import fitz  # PyMuPDF for text extraction
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Download NLTK resources
nltk.download('punkt', quiet=True)

class PDFChatRAG:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2", 
                 response_model="google/flan-t5-base"):
        """
        Initialize the RAG pipeline with embedding and response generation models
        
        Args:
            embedding_model (str): Model for creating text embeddings
            response_model (str): Model for generating responses
        """
        # Embedding model for creating vector representations
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Response generation model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(response_model)
        self.response_model = AutoModelForSeq2SeqLM.from_pretrained(response_model)

    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from a PDF file
        
        Args:
            pdf_path (str): Path to the PDF file
        
        Returns:
            str: Extracted text from the PDF
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def chunk_text(self, text, chunk_size=300, overlap=50):
        """
        Chunk text into smaller segments with optional overlap
        
        Args:
            text (str): Input text to chunk
            chunk_size (int): Maximum size of each chunk
            overlap (int): Number of characters to overlap between chunks
        
        Returns:
            list: List of text chunks
        """
        sentences = sent_tokenize(text)
        chunks, current_chunk = [], ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                # Use overlap to maintain context
                current_chunk = current_chunk[-overlap:] + sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def create_vector_index(self, chunks):
        """
        Create a FAISS index from text embeddings
        
        Args:
            chunks (list): List of text chunks
        
        Returns:
            tuple: FAISS index and embeddings
        """
        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks)
        
        # Convert to numpy array
        embeddings_np = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_np)
        
        return index, embeddings_np

    def retrieve_relevant_chunks(self, query, chunks, index, top_k=3):
        """
        Retrieve most relevant chunks for a given query
        
        Args:
            query (str): User's query
            chunks (list): All text chunks
            index (faiss.Index): FAISS index of embeddings
            top_k (int): Number of top chunks to retrieve
        
        Returns:
            list: Most relevant chunks
        """
        # Embed query
        query_embedding = self.embedding_model.encode([query])[0].astype('float32')
        
        # Search index
        distances, indices = index.search(np.array([query_embedding]), top_k)
        
        return [chunks[i] for i in indices[0]]

    def generate_response(self, query, retrieved_chunks):
        """
        Generate a response using retrieval-augmented generation
        
        Args:
            query (str): User's query
            retrieved_chunks (list): Relevant text chunks
        
        Returns:
            str: Generated response
        """
        # Combine query and context
        context = " ".join(retrieved_chunks)
        prompt = f"Question: {query}\nContext: {context}\nAnswer:"
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", 
                                max_length=512, 
                                truncation=True, 
                                padding=True)
        
        # Generate response
        outputs = self.response_model.generate(
            inputs.input_ids, 
            max_length=200, 
            num_beams=4, 
            early_stopping=True
        )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def chat_with_pdf(self, pdf_path, query):
        """
        Main method to chat with a PDF document
        
        Args:
            pdf_path (str): Path to the PDF file
            query (str): User's query
        
        Returns:
            str: Response to the query
        """
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        
        # Chunk the text
        chunks = self.chunk_text(text)
        
        # Create vector index
        index, _ = self.create_vector_index(chunks)
        
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(query, chunks, index)
        
        # Generate response
        response = self.generate_response(query, relevant_chunks)
        
        return response

# Example usage
def main():
    rag_pipeline = PDFChatRAG()
     
# Prompt the user for the PDF path
    pdf_path = input("Enter the path to your PDF file: ")
    
# Verify that the file exists
    if not os.path.exists(pdf_path):
        print("The specified file does not exist. Please check the path and try again.")
    else:
    # If the file exists, prompt the user for a query
        for i in range(5):# atleast 5 prompts
            query = input("Enter your query about the PDF: ")
            # Process the query using the RAG pipeline
            response = rag_pipeline.chat_with_pdf(pdf_path, query)
            print(f"\nResponse: {response}")

if __name__ == "__main__":
    main()