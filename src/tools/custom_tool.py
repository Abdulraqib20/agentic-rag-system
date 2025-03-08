__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from crewai.tools import BaseTool
from typing import Type, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from markitdown import MarkItDown
from chonkie import SemanticChunker
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import requests

from config.appconfig import FIRECRAWL_API_KEY

# ---------------------------
# DocumentSearchTool Section
# ---------------------------

class DocumentSearchToolInput(BaseModel):
    """Input schema for DocumentSearchTool."""
    query: str = Field(..., description="Query to search the document.")

class DocumentSearchTool(BaseTool):
    name: str = "DocumentSearchTool"
    description: str = "Search the document for the given query."
    args_schema: Type[BaseModel] = DocumentSearchToolInput
    
    model_config = ConfigDict(extra="allow")
    def __init__(self, file_path: str):
        """Initialize the searcher with a PDF file path and set up the Qdrant collection."""
        super().__init__()
        self.file_path = file_path
        self.client = QdrantClient(":memory:")  # For small experiments
        self._process_document()

    def _extract_text(self) -> str:
        """Extract raw text from PDF using MarkItDown."""
        md = MarkItDown()
        result = md.convert(self.file_path)
        return result.text_content

    def _create_chunks(self, raw_text: str) -> list:
        """Create semantic chunks from raw text."""
        chunker = SemanticChunker(
            embedding_model="minishlab/potion-base-8M",
            threshold=0.5,
            chunk_size=512,
            min_sentences=1
        )
        return chunker.chunk(raw_text)

    def _process_document(self):
        """Process the document and add chunks to Qdrant collection."""
        raw_text = self._extract_text()
        chunks = self._create_chunks(raw_text)
        
        docs = [chunk.text for chunk in chunks]
        metadata = [{"source": os.path.basename(self.file_path)} for _ in range(len(chunks))]
        ids = list(range(len(chunks)))

        self.client.add(
            collection_name="demo_collection",
            documents=docs,
            metadata=metadata,
            ids=ids
        )

    def _run(self, query: str) -> list:
        """Search the document with a query string."""
        relevant_chunks = self.client.query(
            collection_name="demo_collection",
            query_text=query
        )
        docs = [chunk.document for chunk in relevant_chunks]
        separator = "\n___\n"
        return separator.join(docs)


# ---------------------------------
# FireCrawlWebSearchTool Section
# ---------------------------------

class FireCrawlWebSearchToolInput(BaseModel):
    """Input schema for FireCrawlWebSearchTool."""
    query:str = Field(..., description='Query to search the web.')
    max_results:int = Field(default=5, description='Maximum number of results to return.')

class FireCrawlWebSearchTool(BaseTool):
    name: str = "FireCrawlWebSearchTool"
    description: str = "Perform a robust web search using the FireCrawl API."
    args_schema: Type[BaseModel] = FireCrawlWebSearchToolInput
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    def __init__(self, api_key: str, base_url: str = "https://api.firecrawl.com/search"):
        """
        Initialize the FireCrawlWebSearchTool with the API key and base URL.
        
        Args:
            api_key (str): Your FireCrawl API key.
            base_url (str): The base URL for the FireCrawl search API.
        """
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
    
    def _run(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web using FireCrawl API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        params = {
            "query": query,
            "max_results": max_results
        }
        retries = 3
        
        for attempt in range(retries):
            try:
                response = requests.get(self.base_url, headers=headers, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                results = data.get("results", [])
                if results:
                    return "\n".join([result.get("snippet", "No snippet available") for result in results])
                else:
                    return "No results found."
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    
                    import time
                    time.sleep(2)  # Pause briefly before retrying
                    continue
                else:
                    return f"Error during FireCrawl web search: {e}"


# Test the implementation
def test_document_searcher():
    # Test file path
    pdf_path = "src/knowledge/00_LATEST_RAQIB_CV.pdf"
    
    # Create instance
    searcher = DocumentSearchTool(file_path=pdf_path)
    
    # Test search
    result = searcher._run("What is the most recent experience of Omotosho")
    print("Search Results:", result)

def test_firecrawl_web_searcher():
    firecrawl_api_key = FIRECRAWL_API_KEY
    
    # Create instance
    searcher = FireCrawlWebSearchTool(api_key=firecrawl_api_key)
    
    # Test search
    result = searcher._run("latest AI trends", max_results=3)
    print("Web Search Results:", result)


if __name__ == "__main__":
    # Test DocumentSearchTool
    test_document_searcher()
    
    # Test FireCrawlWebSearchTool
    test_firecrawl_web_searcher()
