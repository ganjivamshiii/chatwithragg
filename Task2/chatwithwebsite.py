import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
import re

class WebsiteRAGPipeline:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        """
        Initialize the RAG pipeline with embedding and vector database components
        
        :param embedding_model_name: Name of the SentenceTransformer model to use
        """
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize vector database client
        self.chroma_client = chromadb.Client()
        
        # Get or create the collection - this avoids the error if it already exists
        self.collection = self.chroma_client.get_or_create_collection(name="website_content") 
    
    # ... (rest of your class code remains the same) ...
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize webpage text
        
        :param text: Raw text from webpage
        :return: Cleaned text
        """
        # Remove extra whitespaces, newlines, and standardize text
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def scrape_website(self, url: str, max_depth: int = 2) -> List[Dict]:
        """
        Crawl and scrape content from a website
        
        :param url: Website URL to scrape
        :param max_depth: Maximum depth of links to follow
        :return: List of scraped content chunks
        """
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract main content sections
            content_chunks = []
            
            # Extract text from various HTML sections
            text_sections = soup.find_all(['p', 'div', 'article', 'section'])
            
            for section in text_sections:
                text = section.get_text(strip=True)
                if len(text) > 50:  # Only consider substantial text chunks
                    cleaned_text = self.clean_text(text)
                    content_chunks.append({
                        'text': cleaned_text,
                        'source': url
                    })
            
            return content_chunks
        
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return []
    
    def embed_and_store_content(self, content_chunks: List[Dict]):
        """
        Convert content chunks to embeddings and store in vector database
        
        :param content_chunks: List of content chunks to embed and store
        """
        for i, chunk in enumerate(content_chunks):
            # Generate embedding
            embedding = self.embedding_model.encode(chunk['text']).tolist()
            
            # Store in Chroma vector database
            self.collection.add(
                embeddings=[embedding],
                documents=[chunk['text']],
                metadatas=[{'source': chunk['source']}],
                ids=[f"chunk_{i}"]
            )
    
    def query_websites(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve and rank most relevant content for a user query
        
        :param query: User's natural language query
        :param top_k: Number of top results to return
        :return: List of most relevant content chunks
        """
        # Convert query to embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Perform similarity search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return [{
            'text': doc,
            'source': metadata['source']
        } for doc, metadata in zip(results['documents'][0], results['metadatas'][0])]
    
    def generate_response(self, query: str, retrieved_chunks: List[Dict]) -> str:
        """
        Generate a response using retrieved content and LLM
        
        :param query: Original user query
        :param retrieved_chunks: Most relevant content chunks
        :return: Generated response
        """
        # In a real implementation, you'd use an LLM like OpenAI's GPT or Anthropic's Claude
        # Here's a simple placeholder implementation
        context = "\n".join([chunk['text'] for chunk in retrieved_chunks])
        
        prompt = f"""Based on the following context, answer the query:
        
        Query: {query}
        Context: {context}
        
        Response:"""
        
        # Placeholder response generation (replace with actual LLM call)
        return f"Generated response based on {len(retrieved_chunks)} retrieved chunks."
    
    def process_websites(self, urls: List[str]):
        """
        Process multiple websites end-to-end
        
        :param urls: List of website URLs to process
        """
        for url in urls:
            print(f"Processing {url}...")
            content_chunks = self.scrape_website(url)
            self.embed_and_store_content(content_chunks)
        
        print("Website processing complete.")

# Example usage
def main():
    urls = [
        'https://www.uchicago.edu/'
    ]
    
    rag_pipeline = WebsiteRAGPipeline()
    
    # Process websites
    rag_pipeline.process_websites(urls)
    
    # Example query
    query = input("query:")
    retrieved_chunks = rag_pipeline.query_websites(query)
    print("Response:")
    for i in range(len(retrieved_chunks)):
        print(retrieved_chunks[i]['text'])

if __name__ == "__main__":
    main()