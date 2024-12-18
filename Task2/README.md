# Website Research Opportunities RAG Pipeline

## Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline for extracting and querying research opportunity information from university websites. The system uses web scraping, vector embeddings, and intelligent retrieval to help users find relevant research-related information.

## Features

- 🌐 Web Scraping: Crawl multiple university websites
- 🔍 Semantic Search: Convert content to vector embeddings
- 💡 Intelligent Retrieval: Find most relevant information for user queries
- 📄 Contextual Response Generation: Synthesize informative responses

## Prerequisites

- Python 3.8+
- Internet connection for web scraping

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/website-rag-pipeline.git
cd website-rag-pipeline
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- requests
- beautifulsoup4
- numpy
- sentence-transformers
- scikit-learn
- chromadb

## Usage

### Basic Execution

```python
from website_rag_pipeline import WebsiteRAGPipeline

# Initialize the pipeline
rag_pipeline = WebsiteRAGPipeline()

# Define websites to process
urls = [
    "www.url.com"
]

# Process websites
rag_pipeline.process_websites(urls)

# Query for research opportunities
query = "Tell me about research opportunities at the university"
retrieved_chunks = rag_pipeline.query_websites(query)
response = rag_pipeline.generate_response(query, retrieved_chunks)

print(response)
```

## Configuration

- Modify `urls` list to include websites of interest
- Adjust `embedding_model_name` to use different embedding models
- Customize `clean_text()` method for specific text preprocessing needs

## Limitations

- Depends on website structure and accessibility
- Performance varies based on website content
- No guarantee of 100% accurate information retrieval

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


