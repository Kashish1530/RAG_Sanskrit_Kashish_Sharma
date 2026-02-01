import os
import sys
import html
import re

# 1. Suppress Warnings
os.environ["TRANSFORMERS_NO_TF"] = "1"

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Force UTF-8 Encoding
sys.stdout.reconfigure(encoding='utf-8')

# 2. Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "code", "faiss_index")
MODEL_PATH = "./qwen.gguf"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "qwen.gguf")

# ==========================================
# PART 1: REGEX LOGIC (The "Sniper")
# ==========================================
def extract_answer_by_pattern(docs, question):
    question = question.strip()
    text = " ".join([d.page_content for d in docs])

    # Pattern 1: NAME questions
    if re.search(r'नाम\s*(किं|किम्|किमस्ति|किमास्ति|अस्ति)', question, re.IGNORECASE):
        match = re.search(r'(\S+)\s+नाम\s+(\S+)', text)
        if match:
            return f"✓ {match.group(1)} नाम {match.group(2)} अस्ति ।", text

    # Pattern 2: WHO questions
    if re.search(r'(कः|का|के)\s+(आसीत्|अस्ति|आसन्)', question):
        match = re.search(r'(\S+)\s+(आसीत्|अस्ति)', text)
        if match:
            return f"✓ {match.group(1)} {match.group(2)} ।", text

    # Pattern 3: WHERE questions
    if re.search(r'कुत्र\s+(गच्छति|आगच्छति|अस्ति|आसीत्)', question):
        match = re.search(r'(\S+)\s+(गच्छति|आगच्छति|वसति)', text)
        if match:
            return f"✓ {match.group(1)} {match.group(2)} ।", text

    # Pattern 4: DESCRIPTION
    if re.search(r'(कीदृशम्|कीदृश|कीदृशः|कथम्|वातावरणं)', question):
        match = re.search(r'(\S+)\s+(भवति|अस्ति)', text)
        if match:
            return f"✓ {match.group(1)}", text

    return None, None

# ==========================================
# PART 2: LLM LOGIC (The "Scholar")
# ==========================================
def build_llm():
    if not os.path.exists(MODEL_PATH):
        print(f" Error: Model file not found at {MODEL_PATH}")
        sys.exit(1)

    print(f"  Loading LLM (Hybrid Mode)...")
    return LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.1,
        max_tokens=150,
        n_ctx=8192,
        n_gpu_layers=0,
        repeat_penalty=1.3,
        # UPDATED: Added stop tokens to prevent hallucination/rambling
        stop=["।", "\n", "Question:", "Context:", "<|im_end|>"],
        verbose=False
    )

def clean_llm_response(text):
    text = text.strip()
    text = text.replace("है", "अस्ति")
    text = text.replace("यह", "एषः")
    text = text.replace("से", "तः")
    return text

# ==========================================
# MAIN APPLICATION
# ==========================================
def main():
    if not os.path.exists(DB_PATH):
        print(" FAISS index not found. Please run 'ingest.py' first.")
        return

    print("=" * 70)
    print("🕉  SANSKRIT AI SYSTEM (Hybrid: Regex + LLM)")
    print("=" * 70)
    print("\n Loading Search Engine...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

    # UPDATED: More precise retriever (Similarity search with k=2)
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )

    llm = build_llm()

    template = """<|im_start|>system
    You are a Sanskrit Scholar. 
    Your task is to find the SUBJECT (who) and ACTION (what) from the Context.
    Context: {context}
    Question: {question}
    Answer in one short Sanskrit sentence. If the answer is not clear, say "न जानामि" (I don't know).
    <|im_end|>
    <|im_start|>assistant
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    rag_chain = (
            {"context": lambda x: x["context"], "question": lambda x: x["question"]}
            | prompt
            | llm
            | StrOutputParser()
    )

    print(" System Ready!\n")
    print("=" * 70)

    while True:
        q = input("\n प्रश्नः: ").strip()
        if q.lower() in ["exit", "quit", "बाहिर"]:
            break
        if not q: continue
        q = html.unescape(q)

        print(f"Thinking...", end=" ", flush=True)

        try:
            docs = retriever.invoke(q)
            if not docs:
                print("\n No info found.")
                continue

            # TRY REGEX FIRST
            regex_ans, source = extract_answer_by_pattern(docs, q)

            if regex_ans:
                print("\r", end="")
                print("\n" + "=" * 70)
                print(" EXACT MATCH (Regex):")
                print("=" * 70)
                print(f"\n   {regex_ans}\n")
                print("=" * 70)
                continue

            # TRY LLM SECOND
            context_text = "\n\n".join([d.page_content for d in docs])
            raw_response = rag_chain.invoke({"context": context_text, "question": q})
            clean_response = clean_llm_response(raw_response)

            print("\r", end="")
            print("\n" + "=" * 70)
            print(" AI ANSWER (LLM):")
            print("=" * 70)
            print(f"\n   {clean_response}\n")
            print("=" * 70)

        except Exception as e:
            print(f"\n Error: {e}")

if __name__ == "__main__":
    main()