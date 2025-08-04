import pdfplumber
import re

def extract_sections_by_headings(pdf_path):
    # Extract all text from PDF
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
    
    # Define pattern to find top-level headings (e.g., ". Introduction")
    heading_pattern = re.compile(r'^\s*\.\s*([^\n\r]+)', re.MULTILINE)
    heading_matches = list(heading_pattern.finditer(full_text))
    
    if not heading_matches:
        return []
    
    # Extract sections between headings
    sections = []
    for i in range(len(heading_matches)):
        heading = heading_matches[i].group(1).strip()  # Clean heading text
        
        # Determine content boundaries
        start_pos = heading_matches[i].end()
        end_pos = heading_matches[i+1].start() if i < len(heading_matches)-1 else len(full_text)
        
        content = full_text[start_pos:end_pos].strip()
        sections.append((heading, content))
    
    return sections

# Example usage
pdf_path = "sample.pdf"  # Replace with your PDF path
sections = extract_sections_by_headings(pdf_path)

# Print results (as shown in the output sample)
for heading, content in sections:
    print(f"Heading: {heading}")
    print("Text:")
    print(content)
    print("\n" + "-" * 50 + "\n")
