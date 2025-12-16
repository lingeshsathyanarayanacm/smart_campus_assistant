import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
from datetime import datetime
import os
import io
import shutil

if os.path.exists(".env"):
    load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_llm_model():
    return ChatOpenAI(
        model="meta-llama/llama-3.1-8b-instruct",
        temperature=0.3,
        max_tokens=150,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1"
    )

def init_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'generated_content' not in st.session_state:
        st.session_state.generated_content = {}
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None

def get_pdf_text(pdf_docs):
    text = ""
    if not pdf_docs:
        st.warning("Please upload at least one PDF file")
        return ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.warning(f" Could not read {pdf.name}: {str(e)}")
    if not text.strip():
        st.error(" No text extracted from PDFs")
    return text

def get_text_chunks(text):
    if not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=1500)
    return splitter.split_text(text)

def delete_old_index(index_path="faiss_index"):
    if os.path.exists(index_path):
        shutil.rmtree(index_path)
        st.info(" Old FAISS index removed.")

def get_vector_store(text_chunks, index_path="faiss_index"):
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(index_path)
    return vector_store

def load_faiss_index_safe(index_path="faiss_index"):
    embeddings = get_embeddings()
    if not os.path.exists(index_path):
        return None
    try:
        return FAISS.load_local(index_path, embeddings)
    except Exception as e:
        st.warning("FAISS index corrupted. Rebuilding...")
        delete_old_index(index_path)
        return None


def get_conversational_chain():
    prompt_template = """
Answer the question as detailed as possible from the provided context.
If the answer is not in the provided context, say "Answer is not available in the context".

Context:
{context}

Question: {question}

Answer:
"""
    model = get_llm_model()
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return model, prompt

def generate_response(query, vector_store):
    docs = vector_store.similarity_search(query, k=2)
    if not docs:
        return "No relevant information found."
    model, prompt = get_conversational_chain()
    context = "\n\n".join([doc.page_content for doc in docs])
    formatted_prompt = prompt.format(context=context, question=query)
    
    response = model([HumanMessage(content=formatted_prompt)])
    
    if hasattr(response, "content"):
        return response.content
    elif hasattr(response, "generations"):
        return response.generations[0][0].text
    elif isinstance(response, str):
        return response
    else:
        return " Could not parse response"


def summarize_pdf(vector_store):
    summary = generate_response("Summarize main points and key information", vector_store)
    if summary:
        st.session_state.generated_content['summary'] = summary
        st.markdown("### 📋 Summary")
        st.write(summary)

def generate_questions(vector_store):
    questions = generate_response("Generate 5 key questions from the document", vector_store)
    if questions:
        st.session_state.generated_content['questions'] = questions
        st.markdown("### ❓ Questions")
        for i, q in enumerate(questions.split("\n"), 1):
            if q.strip(): st.markdown(f"**{i}.** {q.strip()}")

def generate_mcqs(vector_store):
    mcqs = generate_response("Generate 5 MCQs with options A-D and answers", vector_store)
    if mcqs:
        st.session_state.generated_content['mcqs'] = mcqs
        st.markdown("### 📝 MCQs")
        st.write(mcqs)

def generate_notes(vector_store):
    notes = generate_response("Generate main concepts and key points", vector_store)
    if notes:
        st.session_state.generated_content['notes'] = notes
        st.markdown("### 📚 Notes")
        st.write(notes)

def create_download_link(content, filename, label):
    buffer = io.BytesIO()
    buffer.write(content.encode())
    buffer.seek(0)
    st.download_button(label=label, data=buffer, file_name=filename, mime="text/plain")


def main():
    st.set_page_config(page_title="Smart Campus Assistant", page_icon="🤖", layout="wide")
    init_session_state()
    st.markdown("<h1 style='text-align:center'>Smart Campus Assistant</h1>", unsafe_allow_html=True)


    with st.sidebar:
        st.markdown("### 📁 Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])
        if pdf_docs: st.success(f"✅Uploaded {len(pdf_docs)} PDFs")
        if st.button("⚙️ Process Documents"):
            if not pdf_docs:
                st.error("Upload at least one PDF")
            else:
                text = get_pdf_text(pdf_docs)
                if text:
                    chunks = get_text_chunks(text)
                    if chunks:
                        vector_store = get_vector_store(chunks)
                        st.session_state.vector_store = vector_store
                        st.success("✅ Documents processed successfully!")

  
    vector_store = st.session_state.get("vector_store")
    if vector_store:
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.button("📋 Summarize PDF", on_click=summarize_pdf, args=(vector_store,))
        with col2: st.button("❓ Generate Questions", on_click=generate_questions, args=(vector_store,))
        with col3: st.button("📝 Create MCQs", on_click=generate_mcqs, args=(vector_store,))
        with col4: st.button("📚 Generate Notes", on_click=generate_notes, args=(vector_store,))

        st.markdown("---")
        user_question = st.text_input("Ask a question about your PDFs")
        if user_question:
            answer = generate_response(user_question, vector_store)
            st.session_state.chat_history.append({'question': user_question, 'answer': answer, 'timestamp': datetime.now()})
            st.success("✅ Answer found!")
            st.write(answer)

        if st.session_state.generated_content:
            st.markdown("### 📥 Download Generated Content")
            for key, content in st.session_state.generated_content.items():
                create_download_link(content, f"{key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", f"📥 Download {key.capitalize()}")

if __name__ == "__main__":
    main()
