from typing import List
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
    PDFMinerLoader,
    UnstructuredWordDocumentLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
import os
from pathlib import Path

class FileProcessor:
    def __init__(self, persist_directory: str = "chroma_db"):
        """Initialize the FileProcessor with ChromaDB storage location.
        
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
        
    def load_file(self, file_path: str) -> List[str]:
        """Load and process different file types.
        
        Args:
            file_path (str): Path to the file to be processed
            
        Returns:
            List[str]: List of documents/pages from the file
        """
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == '.pdf':
                loader = PDFMinerLoader(file_path)
            elif file_extension == '.docx':
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_extension == '.txt':
                loader = TextLoader(file_path)
            else:
                # For other file types, try using Unstructured
                loader = UnstructuredFileLoader(file_path)
                
            documents = loader.load()
            return documents
            
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

    def process_and_store(self, file_path: str, 
                         collection_name: str = "documents") -> Chroma:
        """Process file and store in ChromaDB.
        
        Args:
            file_path (str): Path to the file to be processed
            collection_name (str): Name of the ChromaDB collection
            
        Returns:
            Chroma: ChromaDB vector store instance
        """
        # Load the document
        documents = self.load_file(file_path)
        
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
