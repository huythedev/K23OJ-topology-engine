import argparse
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from database import Database
from processor import Processor
from engine import Engine

def process_one_problem(problem):
    """
    Helper function to process a single problem (download PDF, extract text).
    Returns list of chunks if successful.
    """
    try:
        text = Processor.process_problem(problem)
        if not text or len(text) < 10: # Skip empty or too short
            return None
        
        # Chunk the text to handle large PDFs
        chunks = Processor.chunk_text(text)
        
        if not chunks:
            return None

        # Return list of items to add
        return {
            "chunks": chunks,
            "metadata": {
                "code": problem.get('code', 'unknown'),
                "name": problem.get('name', 'unknown'),
                "id": str(problem.get('id', 'unknown'))
            },
            "id": str(problem.get('id'))
        }
    except Exception as e:
        print(f"Error processing problem {problem.get('id')}: {e}")
        return None

def cmd_index(args):
    db = Database()
    try:
        print("Fetching problems from database...")
        problems = db.fetch_all_problems()
        print(f"Found {len(problems)} problems.")
    except Exception as e:
        print(f"Error fetching problems: {e}")
        return
    finally:
        db.close()

    engine = Engine()
    
    # Clear existing index before starting (to avoid duplicates)
    engine.reset_index()

    batch_size = 32
    contents = []
    metadatas = []
    ids = []
    
    # Use multiple threads to download PDFs in parallel.
    # I/O bound task, so threads are very effective.
    max_workers = int(os.getenv("MAX_WORKERS", 20))
    
    print(f"Processing and indexing problems using {max_workers} threads...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_prob = {executor.submit(process_one_problem, p): p for p in problems}
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_prob), total=len(problems)):
            result = future.result()
            if result:
                # Add all chunks for this problem
                for chunk in result["chunks"]:
                    contents.append(chunk)
                    metadatas.append(result['metadata'])
                    ids.append(result['id'])
            
            # Batch add to engine to avoid holding everything in memory if dataset was huge
            # (though for 800 problems it fits easily, this pattern scales better)
            if len(contents) >= batch_size:
                engine.add_problems(contents, metadatas, ids)
                contents = []
                metadatas = []
                ids = []

    # Add remaining
    if contents:
        engine.add_problems(contents, metadatas, ids)
    
    # Save the index once at the end!
    engine.save_index()
    print("Indexing complete.")

def cmd_check(args):
    engine = Engine()
    
    query_text = ""
    
    if args.file:
        if args.file.endswith('.pdf'):
            # Simulate a problem dict with pdf_url (local file handling would need tweak in processor, but let's assume it's text for now or simple read)
            # Actually processor expects URL for PDF. Let's add local read support here or just read text.
            from pypdf import PdfReader
            try:
                reader = PdfReader(args.file)
                for page in reader.pages:
                    query_text += page.extract_text() + " "
            except Exception as e:
                print(f"Error reading PDF file: {e}")
                return
        else:
            with open(args.file, 'r', encoding='utf-8') as f:
                query_text = f.read()
                # If it's markdown, clean it
                query_text = Processor.extract_text_from_markdown(query_text)
    elif args.text:
        query_text = args.text
    else:
        print("Please provide --file or --text")
        return

    if not query_text.strip():
        print("Empty query text.")
        return

    print("Searching for duplicates...")
    results = engine.search(query_text, n_results=5)
    
    # ChromaDB results structure:
    # {'ids': [['id1', ...]], 'distances': [[0.2, ...]], 'metadatas': [[{'name':...}, ...]], 'documents': ...}
    
    if results and results['ids']:
        ids = results['ids'][0]
        distances = results['distances'][0]
        metas = results['metadatas'][0]
        
        print("\nTop 5 Similar Problems:")
        seen_ids = set()
        count = 0
        for i in range(len(ids)):
            p_id = ids[i]
            if p_id in seen_ids:
                continue
            seen_ids.add(p_id)
            
            similarity = (1 - distances[i]) * 100
            print(f"{count+1}. [{metas[i].get('code')}] {metas[i].get('name')} (Similarity: {similarity:.2f}%)")
            count += 1
            if count >= 5:
                break
    else:
        print("No results found.")

def main():
    parser = argparse.ArgumentParser(description="K23OJ Topology Engine - Duplicate Detection")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Index command
    parser_index = subparsers.add_parser("index", help="Index all problems from DB")
    
    # Check command
    parser_check = subparsers.add_parser("check", help="Check for duplicates")
    parser_check.add_argument("--file", help="Path to local file (md/txt/pdf) containing new problem")
    parser_check.add_argument("--text", help="Raw text string of new problem")
    
    args = parser.parse_args()
    
    if args.command == "index":
        cmd_index(args)
    elif args.command == "check":
        cmd_check(args)

if __name__ == "__main__":
    main()
