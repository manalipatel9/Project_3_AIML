import os
import gradio as gr
from typing import List, Union
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from rag.file_processor import FileProcessor
from rag.url_processor import URLProcessor
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import shutil
import time

class RAGApplication:
    def __init__(self, persist_directory: str = "chroma_db"):
        self.persist_directory = persist_directory
        self.file_processor = FileProcessor(persist_directory=persist_directory)
        self.url_processor = URLProcessor(persist_directory=persist_directory)
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )
        self.vectorstore = None
        self.chain = None
        self.current_collection = None
        
        # Ensure the directory exists
        os.makedirs(self.persist_directory, exist_ok=True)

    def initialize_chain(self):
        if self.vectorstore is None:
            raise ValueError("Please load some documents first!")
            
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(),
            memory=self.memory,
            return_source_documents=True,
            return_generated_question=True,
            combine_docs_chain_kwargs={"prompt": None},
            verbose=True
        )

    def process_file(self, file_path: str) -> str:
        try:
            # Create a unique collection name for this upload
            collection_name = f"file_{os.path.basename(file_path)}_{int(time.time())}"
            self.vectorstore = self.file_processor.process_and_store(
                file_path=file_path,
                collection_name=collection_name
            )
            self.current_collection = collection_name
            self.initialize_chain()
            return f"Successfully processed file: {file_path}"
        except Exception as e:
            return f"Error processing file: {str(e)}"

    def process_url(self, url: str) -> str:
        try:
            # Create a unique collection name using timestamp and URL length
            collection_name = f"url_{len(url)}_{int(time.time())}"
            self.vectorstore = self.url_processor.process_and_store(
                url=url,
                collection_name=collection_name
            )
            self.current_collection = collection_name
            self.initialize_chain()
            return f"Successfully processed URL: {url}"
        except Exception as e:
            return f"Error processing URL: {str(e)}"

    def process_urls(self, urls: List[str]) -> str:
        try:
            # Create a unique collection name for this upload
            collection_name = f"urls_batch_{int(time.time())}"
            self.vectorstore = self.url_processor.process_multiple_urls(
                urls=urls,
                collection_name=collection_name
            )
            self.current_collection = collection_name
            self.initialize_chain()
            return f"Successfully processed {len(urls)} URLs"
        except Exception as e:
            return f"Error processing URLs: {str(e)}"

    def query(self, question: str) -> str:
        if self.chain is None:
            return "Please load some documents first!"
        
        try:
            result = self.chain({"question": question})
            return result["answer"]
        except Exception as e:
            return f"Error processing query: {str(e)}"

    def clear_memory_and_storage(self) -> str:
        """Clear conversation memory and current documents."""
        try:
            # Clear conversation memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                output_key="answer",
                return_messages=True
            )
            
            # Clear current references
            self.vectorstore = None
            self.chain = None
            
            # If there's a current collection, delete it
            if self.current_collection:
                collection_path = os.path.join(self.persist_directory, self.current_collection)
                if os.path.exists(collection_path):
                    shutil.rmtree(collection_path)
                self.current_collection = None
            
            return "Successfully cleared current documents and chat history"
        except Exception as e:
            return f"Error clearing data: {str(e)}"

def create_gradio_interface():

    rag_app = RAGApplication()

    def process_input(input_type: str, input_value: Union[str, List[str]]) -> str:
        if input_type == "file":
            return rag_app.process_file(input_value)
        elif input_type == "url":
            return rag_app.process_url(input_value)
        elif input_type == "multiple_urls":
            urls = [url.strip() for url in input_value.split(',')]
            return rag_app.process_urls(urls)
        return "Invalid input type"

    def chat(message: str, history: List[List[str]]) -> str:
        return rag_app.query(message)

    with gr.Blocks() as demo:
        gr.Markdown("# RAG Application")
        
        with gr.Tab("Load Data"):
            input_type = gr.Radio(
                choices=["file", "url", "multiple_urls"],
                label="Input Type",
                value="file"
            )
            
            input_value = gr.Textbox(
                label="Input (file path, URL, or comma-separated URLs)",
                lines=2
            )
            
            process_btn = gr.Button("Process Input")
            clear_btn = gr.Button("Clear All Data", variant="secondary")
            process_output = gr.Textbox(label="Processing Result")
            
            process_btn.click(
                fn=process_input,
                inputs=[input_type, input_value],
                outputs=process_output
            )
            
            clear_btn.click(
                fn=lambda: rag_app.clear_memory_and_storage(),
                inputs=[],
                outputs=process_output
            )

        with gr.Tab("Chat"):
            chatbot = gr.ChatInterface(
                fn=chat,
                title="Chat with your documents",
                description="Ask questions about the loaded documents"
            )

    return demo

if __name__ == "__main__":
    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")
    
    # Launch the Gradio interface
    demo = create_gradio_interface()
    demo.launch(share = True, server_name="0.0.0.0")
