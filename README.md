# RAG (Retrieval-Augmented Generation) Application

A powerful document question-answering system that combines the capabilities of Large Language Models with local document processing. This application allows users to load documents or web content and engage in meaningful conversations about their content.

### Group Members
* Matthew Dlugosz
* Katia Monteros
* Manali Patel
* Gita Rani 

#### What is RAG?
'Retrieval Augmented Generation (RAG) is a method that improves the performance of Large Language Models (LLMs) by integrating an information retrieval component with the model's text generation capabilities.'

<div align="center">
<img src="https://www.clarifai.com/hs-fs/hubfs/rag-query-drawio%20(1)-png-2.png?width=2056&height=1334&name=rag-query-drawio%20(1)-png-2.png" alt="Alt Text" width="600" height="400">
</div>

### Features

- Document Processing:
  - Support for local file uploads
  - URL content processing
  - Batch processing of multiple URLs
- Interactive Chat Interface:
  - Natural language conversations about loaded documents
  - Context-aware responses
  - Conversation memory
- Built with modern technologies:
  - LangChain for document processing and LLM integration
  - ChromaDB for vector storage
  - OpenAI's GPT-3.5 Turbo for natural language understanding
  - Gradio for the user interface

### Prerequisites

- Python 3.x
- OpenAI API Key

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd Project_3_AIML
```

2. Create and activate a virtual environment:
```bash
python3 -m venv env
source env/bin/activate  # On Windows, use: env\Scripts\activate
```

3. Install required packages:
```bash
pip install gradio langchain langchain-openai langchain-community chromadb python-dotenv openai unstructured pdfminer.six python-docx
```

4. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

### Chatbot Usage

1. Start the application:
```bash
python main.py
```

2. Access the web interface:
   - The application will start a local server
   - Open your web browser and navigate to the provided URL

3. Using the application:
   - In the "Load Data" tab:
     - Select input type (file, URL, or multiple URLs)
     - Enter the source(s)
     - Click "Process Input" to load the content
   - In the "Chat" tab:
     - Input a question about the loaded documents
     - Receive context-aware responses
     - Continue asking questions or exit the conversation

### Project Structure

- `main.py`: Core application logic and Gradio interface
- `rag/`: Directory containing RAG processing modules
- `chroma_db/`: Vector database storage
- `.env`: Environment variables configuration
- `install.txt`: Installation requirements

### Notes

- The application uses OpenAI's GPT-3.5 Turbo model by default
- Documents are processed and stored locally using ChromaDB
- Conversation history is maintained during each session
- Use the "Clear All Data" button to reset the application state

### Conclusion



### Future Development Plan



### Slideshow



### Citations
https://www.clarifai.com/blog/what-is-rag-retrieval-augmented-generation

