import os
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path, output_txt_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Successfully extracted text to {output_txt_path}")
    except Exception as e:
        print(f"Error extracting text: {e}")

if __name__ == "__main__":
    pdf_path = "250502648v2.pdf"
    output_path = "paper_content.txt"
    extract_text_from_pdf(pdf_path, output_path)
