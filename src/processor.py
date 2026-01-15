import requests
import io
from pypdf import PdfReader
from bs4 import BeautifulSoup
import markdown
import re

class Processor:
    @staticmethod
    def extract_text_from_markdown(md_content: str) -> str:
        """
        Converts markdown to plain text by rendering to HTML then stripping tags.
        """
        if not md_content:
            return ""
        
        # Convert markdown to html
        html = markdown.markdown(md_content)
        
        # Use BS4 to extract text
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=" ")
        
        # Basic cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def extract_text_from_pdf_url(pdf_url: str) -> str:
        """
        Downloads PDF from URL and extracts text.
        """
        if not pdf_url:
            return ""

        # Fix missing domain for relative paths
        if pdf_url.startswith("/"):
            pdf_url = "https://cdn.k23oj.io.vn/media" + pdf_url
            
        try:
            response = requests.get(pdf_url, timeout=10)
            if response.status_code == 200:
                f = io.BytesIO(response.content)
                reader = PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + " "
                
                # Basic cleanup
                text = re.sub(r'\s+', ' ', text).strip()
                return text
            else:
                print(f"Failed to download PDF {pdf_url}: Status {response.status_code}")
                return ""
        except Exception as e:
            print(f"Error processing PDF {pdf_url}: {e}")
            return ""

    @staticmethod
    def process_problem(problem: dict) -> str:
        """
        Determines whether to use description or PDF and returns processed text.
        """
        text = ""
        # Prioritize PDF if description is empty or very short, or just concatenate both?
        # Usually DMOJ problems have description OR pdf.
        
        if problem.get('description'):
            text += Processor.extract_text_from_markdown(problem['description']) + " "
            
        if problem.get('pdf_url'):
            # Only fetch PDF if we don't have enough text or strict requirement.
            # But duplicate checking is better with more data.
            # Let's fetch it.
            pdf_text = Processor.extract_text_from_pdf_url(problem['pdf_url'])
            text += pdf_text
            
        return text.strip()

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
        """
        Splits text into overlapping chunks to handle large documents (like PDFs with multiple problems).
        Targeting ~256 tokens for embeddings (approx 1000 chars is a safe upper bound).
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            # Determine end of current chunk
            end = min(start + chunk_size, text_len)
            
            # If we are not at the end of the text, try to find a natural break point
            if end < text_len:
                # Look for the last period or newline in the second half of the chunk
                search_start = start + int(chunk_size * 0.5)
                # Slice the relevant part to search
                search_area = text[search_start:end]
                
                last_period = search_area.rfind('.')
                last_newline = search_area.rfind('\n')
                
                break_offset = max(last_period, last_newline)
                
                if break_offset != -1:
                    # Adjust end to the break point
                    end = search_start + break_offset + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start forward, respecting overlap (unless we hit the end)
            if end == text_len:
                break
                
            start = max(start + 1, end - overlap)
            
        return chunks
