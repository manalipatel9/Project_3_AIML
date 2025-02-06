from typing import List
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
    PDFMinerLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    WebBaseLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
import os
from pathlib import Path
from urllib.parse import urlparse

class Processor:
    def __init__(self, persist_directory: str = "chroma_db"):
        """Initialize the Processor with ChromaDB storage location.
        
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
        """Check if the URL is valid.
        
        Args:
            url (str): URL to validate
            
        Returns:
            bool: True if URL is valid, False otherwise
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    def load_source(self, source: str) -> List[str]:
        """Load and process content from a file or URL.
        
        Args:
            source (str): Path to file or URL to process
            
        Returns:
            List[str]: List of documents from the source
        """
        if self.is_valid_url(source):
            try:
                loader = WebBaseLoader(source)
                return loader.load()
            except Exception as e:
                raise Exception(f"Error loading URL: {str(e)}")
        
        # Handle as file if not URL
        file_extension = Path(source).suffix.lower()
        try:
            if file_extension == '.pdf':
                loader = PDFMinerLoader(source)
            elif file_extension == '.docx':
                loader = UnstructuredWordDocumentLoader(source)
            elif file_extension == '.txt':
                loader = TextLoader(source)
            else:
                # For other file types, try using Unstructured
                loader = UnstructuredFileLoader(source)
                
            return loader.load()
            
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")

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

    def process_and_store(self, source: str, 
                         collection_name: str = "documents") -> Chroma:
        """Process content from file or URL and store in ChromaDB.
        
        Args:
            source (str): Path to file or URL to process
            collection_name (str): Name of the ChromaDB collection
            
        Returns:
            Chroma: ChromaDB vector store instance
        """
        # Load the content
        documents = self.load_source(source)
        
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

    def process_multiple_sources(self, sources: List[str], 
                               collection_name: str = "documents") -> Chroma:
        """Process multiple files/URLs and store them in the same collection.
        
        Args:
            sources (List[str]): List of file paths or URLs to process
            collection_name (str): Name of the ChromaDB collection
            
        Returns:
            Chroma: ChromaDB vector store instance
        """
        all_chunks = []
        
        for source in sources:
            try:
                documents = self.load_source(source)
                chunks = self.chunk_text(documents)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing source {source}: {str(e)}")
                continue
        
        if not all_chunks:
            raise Exception("No valid content was extracted from the sources")
        
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