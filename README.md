# K23OJ Topology Engine (Problem Duplicate Checker)

A high-performance tool for detecting duplicate programming problems in an Online Judge (DMOJ based) using semantic analysis.

Unlike simple text matching, this engine uses **Sentence Transformers** (Deep Learning) to understand the *meaning* of problem statements. It can identify duplicates even if the wording is different, processing both Markdown descriptions and PDF attachments.

## 🚀 Features

- **Semantic Search**: Finds problems with similar logic/ideas, not just identical text.
- **Multi-Format Support**: Automatically extracts text from **Markdown** descriptions and **PDF** URLs.
- **High Performance**: 
  - Uses `intfloat/multilingual-e5-small` for fast, accurate, and multilingual (including Vietnamese) embeddings.
  - Multi-threaded indexing (configurable `MAX_WORKERS`) to download and process PDFs in parallel.
  - **GPU Acceleration**: Supports Nvidia CUDA and AMD DirectML (RX 580 etc.) for faster processing.
  - Custom in-memory vector engine (replacing heavy vector DBs) optimized for speed and compatibility.
- **Database Integration**: Connects directly to DMOJ's MySQL/PostgreSQL database to fetch problem data.

## 🛠️ Prerequisites

- **Python 3.9+** (Tested on Python 3.14)
- **Database Access**: Read access to the `onlinejudge` database.

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/huythedev/K23OJ-topology-engine.git
   cd K23OJ-topology-engine
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Configuration

1. Create a `.env` file in the root directory (copied from `.env.example`):
   ```bash
   cp .env.example .env
   ```
2. Configure the following variables:
   ```env
   # Database Configuration
   DB_TYPE=mysql
   DB_HOST=localhost
   DB_PORT=3306
   DB_USER=root
   DB_PASS=secret
   DB_NAME=onlinejudge

   # Engine Settings
   MAX_WORKERS=20       # Number of threads for downloading PDFs
   USE_GPU=true         # Set to true to enable CUDA or DirectML (AMD)
   ```

## 🖥️ GPU Support (Optional)

This engine supports GPU acceleration for faster indexing and checking.

- **Nvidia Users**: Install standard PyTorch with CUDA support.
- **AMD Users (Windows)**: Install `torch-directml` for acceleration on cards like RX 580.
  ```bash
  pip install torch-directml
  ```
  Ensure `USE_GPU=true` is set in your `.env`.

2. Edit `.env` with your database credentials:
   ```ini
   DB_TYPE=mysql
   DB_HOST=localhost
   DB_PORT=3306
   DB_USER=root
   DB_PASS=yourpassword
   DB_NAME=yourdbname
   MAX_WORKERS=20
   ```

## 🏃 Usage

### 1. Indexing Problems
Fetch all problems from the database, download PDFs, generate embeddings, and save the index locally.

```bash
python src/main.py index
```
*Note: The index is saved to `data/index.pkl`. Run this periodically when new problems are added to the DB.*

### 2. Checking for Duplicates
Check if a new problem (from a file or raw text) already exists in the database.

**From a file (Markdown, Text, or PDF)**:
```bash
python src/main.py check --file "path/to/new_problem.pdf"
```

**From raw text**:
```bash
python src/main.py check --text "Given a graph with N nodes and M edges..."
```

## 🏗️ Project Structure

```
.
├── src/
│   ├── main.py         # CLI Entry Point & Threading Logic
│   ├── database.py     # Database Connector (MySQL/PostgreSQL)
│   ├── processor.py    # Text Extraction (Markdown/PDF/HTML)
│   └── engine.py       # Vector Engine (SentenceTransformers & Numpy)
├── data/               # Stores the generated index (created at runtime)
├── .env.example        # Configuration template
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```

## 🔧 Technical Details
- **Embedding Model**: Uses `intfloat/multilingual-e5-small` for fast, accurate, and multilingual (including Vietnamese) embeddings.
- **Similarity Metric**: Cosine Similarity.
- **Storage**: Serialized Numpy arrays (via Pickle) for maximum portability and compatibility with newer Python versions where extensive vector DB libraries might have issues.
