from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import os

class URLProcessor:
    def __init__(self, persist_directory: str = "chroma_db"):
        """Initialize the URLProcessor with ChromaDB storage location.
        
        Args:
            persist_directory (str): Directory where ChromaDB will store its data
        """
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        
        # Ensure the directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory
        )
    
    def is_valid_url(self, url: str) -> bool:
        """Check if the URL is valid and accessible.
        
        Args:
            url (str): URL to validate
            
        Returns:
            bool: True if URL is valid and accessible, False otherwise
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def load_url(self, url: str) -> List[str]:
        """Load and process content from a URL.
        
        Args:
            url (str): URL to process
            
        Returns:
            List[str]: List of documents from the URL
        """
        if not self.is_valid_url(url):
            raise ValueError("Invalid URL provided")
            
        try:
            # Use LangChain's WebBaseLoader
            loader = WebBaseLoader(url)
            documents = loader.load()
            return documents
            
        except Exception as e:
            raise Exception(f"Error loading URL: {str(e)}")

    def chunk_text(self, documents: List[str], 
                  chunk_size: int = 1000,
                  chunk_overlap: int = 200) -> List[str]:
        """Split documents into smaller chunks.
        
        Args:
            documents (List[str]): List of documents to be chunked
            chunk_size (int): Size of each chunk
            chunk_overlap (int): Overlap between chunks
            
        Returns:
            List[str]: List of text chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        
        chunks = text_splitter.split_documents(documents)
        return chunks

    def process_and_store(self, url: str, 
                         collection_name: str = "url_documents") -> Chroma:
        """Process URL content and store in ChromaDB.
        
        Args:
            url (str): URL to process
            collection_name (str): Name of the ChromaDB collection
            
        Returns:
            Chroma: ChromaDB vector store instance
        """
        # Load the URL content
        documents = self.load_url(url)
        
        # Chunk the text
        chunks = self.chunk_text(documents)
        
        # Create and store embeddings in ChromaDB
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=collection_name
        )
        
        # Persist the database
        vectorstore.persist()
        
        return vectorstore

    def process_multiple_urls(self, urls: List[str], 
                            collection_name: str = "url_documents") -> Chroma:
        """Process multiple URLs and store them in the same collection.
        
        Args:
            urls (List[str]): List of URLs to process
            collection_name (str): Name of the ChromaDB collection
            
        Returns:
            Chroma: ChromaDB vector store instance
        """
        all_chunks = []
        
        for url in urls:
            try:
                documents = self.load_url(url)
                chunks = self.chunk_text(documents)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing URL {url}: {str(e)}")
                continue
        
        if not all_chunks:
            raise Exception("No valid content was extracted from the URLs")
        
        # Create and store embeddings in ChromaDB
        vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=collection_name
        )
        
        # Persist the database
        vectorstore.persist()
        
        return vectorstore
