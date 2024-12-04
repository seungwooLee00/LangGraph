
# Chatbot System README

## Overview

This project implements a modular chatbot designed to assist users, particularly foreign students, in adapting to university life. It provides features like course recommendations, advice on school-related queries, user-specific information, and general Q&A functionalities. The chatbot leverages advanced AI models, custom workflows, and state management for efficient interaction handling.

---

## Features

- **Intent Analysis**: Determines user intent (`recommendation`, `advice`, `information`, `general`) based on input.
- **Course Recommendations**: Suggests courses based on user queries and provided context.
- **Advice**: Offers guidance on administrative or academic support.
- **User Information Management**: Tracks and retrieves user-specific academic data.
- **General Query Handling**: Responds to generic or ambiguous questions.
- **Document Retrieval**: Uses a multi-query retriever for mixed-language document retrieval.
- **Workflow Orchestration**: A graph-based workflow ensures modular and extendable processing.

---

## Project Structure

```
LangGraph
├── Chatbot                 
│   ├── __init__.py
│   ├── chains.py           # Chains for various chatbot functionalities
│   ├── chatbot.py          # Main chatbot class implementation
│   ├── loader.py           # Document loading utilities for PDFs, CSVs, Markdown
│   └── retriever.py        # Document retriever setup
├── Data                    # Retrieval Dataset
├── .env                    # Environment variables (API keys, configurations)
├── requirements.txt        # Project dependencies
├── sample.py               # sample code
├── test_results.txt        # test result file
└── README.md               # Project documentation
```

---

## Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PORTKEY_API_KEY=your_portkey_api_key
   ```

---

## Usage

### Initialize Chatbot
The `Chatbot` class initializes models, chains, and workflows.

```python
from chatbot import Chatbot

bot = Chatbot()
```

### Query the Chatbot
Use the `ainvoke` method to process queries:
```python
response = await bot.ainvoke("What courses are recommended for AI?", session_id="1234")
print(response)
```

### Workflow
- **Analyze**: Determines user intent.
- **Chains**: Executes the corresponding functionality (`recommendation`, `advice`, etc.).
- **Routing**: Directs input to appropriate chain based on intent.

---

## Key Components

### Models
- **Primary Model**: `gpt-4o` for complex queries.
- **Mini Model**: `gpt-4o-mini` for lightweight tasks.

### Chains
- **Analyze Chain**: Determines intent from input.
- **Advice Chain**: Provides administrative or academic guidance.
- **Recommendation Chain**: Generates course recommendations.
- **User Info Chain**: Retrieves and updates user-specific information.
- **General Chain**: Handles generic queries.

### Retriever
- Multi-query retriever supports English and Korean for mixed-language data.
- Uses Chroma vector store for document embedding and storage.

### Loader
- Reads documents (PDF, CSV, Markdown) for retrieval.

---

## Extensibility

The chatbot architecture allows easy integration of new functionalities by:
1. Adding new chains in `chains.py`.
2. Updating the workflow in `_setup_workflow`.
3. Expanding the `StateGraph` as needed.

---

## Logging

Logs are configured to provide debug-level insights:
- Log file: `chatbot.log`
- Format: `%(asctime)s - %(levelname)s - %(message)s`

---

## Dependencies

Major libraries:
- [LangChain](https://langchain.com)
- [LangGraph](https://langgraph.com)
- [OpenAI](https://openai.com)
- [Portkey AI](https://portkey.ai)

---

## License

This project is licensed under the MIT License. See `LICENSE` for more details.

---

## Contact

For queries or contributions, reach out to [robert032626@gmail.com].
