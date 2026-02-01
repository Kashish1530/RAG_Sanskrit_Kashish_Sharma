import os
import sys
import shutil
import stat
import time
import glob

# Force UTF-8 for Sanskrit characters in terminal
sys.stdout.reconfigure(encoding='utf-8')

from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(BASE_DIR, "code", "faiss_index")


# --- 1. FORCE DELETE HANDLER ---
def on_rm_error(func, path, exc_info):
    """Unlocks files so Windows can delete them."""
    os.chmod(path, stat.S_IWRITE)
    try:
        func(path)
    except Exception:
        pass


def load_txt_files_safely(directory):
    """
    Tries to load .txt files with UTF-8.
    If that fails (like your 0xff error), it retries with UTF-16.
    """
    docs = []
    txt_files = glob.glob(os.path.join(directory, "*.txt"))

    for file_path in txt_files:
        try:
            # Try Standard UTF-8
            loader = TextLoader(file_path, encoding='utf-8')
            docs.extend(loader.load())
            print(f" Loaded (UTF-8): {os.path.basename(file_path)}")
        except Exception:
            try:
                # If failed, Try UTF-16 (Common for Windows Sanskrit files)
                loader = TextLoader(file_path, encoding='utf-16')
                docs.extend(loader.load())
                print(f"Loaded (UTF-16): {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Failed to load {os.path.basename(file_path)}: {e}")
    return docs


def main():
    # --- 2. CLEANUP OLD INDEX ---
    if os.path.exists(DB_PATH):
        print(" Removing old FAISS index...")
        shutil.rmtree(DB_PATH, onerror=on_rm_error)

        # Safety wait for Windows file lock
        if os.path.exists(DB_PATH):
            time.sleep(1)
            try:
                shutil.rmtree(DB_PATH, onerror=on_rm_error)
            except:
                pass

    print(f" Loading Sanskrit documents from {DATA_DIR} ...")

    documents = []

    # --- 3. SMART LOADING ---
    # Load .txt files with auto-encoding detection
    documents.extend(load_txt_files_safely(DATA_DIR))

    # Load .docx files (if any)
    if any(f.endswith(".docx") for f in os.listdir(DATA_DIR)):
        try:
            loader_docx = DirectoryLoader(DATA_DIR, glob="*.docx", loader_cls=Docx2txtLoader)
            documents.extend(loader_docx.load())
            print("Loaded .docx files")
        except Exception as e:
            print(f"Warning loading docx: {e}")

    if not documents:
        print(" Error: No documents successfully loaded.")
        return

    print(f" Successfully loaded {len(documents)} documents.")

    print(" Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    print(f" Created {len(chunks)} chunks.")

    print("Creating embeddings (MiniLM)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print("Building FAISS index...")
    try:
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(DB_PATH)
        print(f" Success! Database saved to: {DB_PATH}")
    except Exception as e:
        print(f" Error saving index: {e}")


if __name__ == "__main__":
    main()