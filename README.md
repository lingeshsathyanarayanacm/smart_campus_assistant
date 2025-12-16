# Smart Campus Assistant
### Developed by **Lingesh Sathya Narayana C.M**

An intelligent campus assistant that helps students by summarizing notes, generating MCQs, extracting keywords, creating explanations, and answering questions—powered by LLMs (OpenAI/OpenRouter).

## Features
- PDF/Text Summarization
- Keyword Extraction
- Question & Answer Generation
- MCQ Generator (randomized + contextual)
- Topic Explanation
- Fast & Lightweight Python Backend
- Works with OpenAI API or OpenRouter API

## Tech Stack
- Python 3.10+
- OpenAI / OpenRouter API
- FAISS (optional for vector storage)
- Streamlit (optional UI)
- FastAPI (if deployed as API)

## Installation
### Clone the Repository
```
git clone https://github.com/lingeshsathyanarayanacm/smart-campus-assistant.git
cd smart-campus-assistant
```

### Create Virtual Environment
```
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### Install Requirements
```
pip install -r requirements.txt
```

## Environment Variables
Create `.env` file:
```
OPENAI_API_KEY=your_key_here
# or
OPENROUTER_API_KEY=your_key_here
```

## Run the Project
### Python App
```
python main.py
```

### Streamlit
```
streamlit run app.py
```

## Deployment (Render.com)
1. Push code to GitHub  
2. Create Web Service on Render  
3. Start Command:
```
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```
or
```
python main.py
```
4. Add environment variables  
5. Render auto-installs from requirements.txt

## Generate requirements.txt
```
pip freeze > requirements.txt
```

## Developer
**Lingesh Sathya Narayana C.M**
AI | ML | Automation Enthusiast
