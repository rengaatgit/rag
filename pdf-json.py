import pdfplumber
import re
import json

def extract_sections_by_headings(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
    
    heading_pattern = re.compile(r'^\s*\.\s*([^\n\r]+)', re.MULTILINE)
    heading_matches = list(heading_pattern.finditer(full_text))
    
    if not heading_matches:
        return []
    
    sections = []
    for i in range(len(heading_matches)):
        heading = heading_matches[i].group(1).strip()
        start_pos = heading_matches[i].end()
        end_pos = heading_matches[i+1].start() if i < len(heading_matches)-1 else len(full_text)
        content = full_text[start_pos:end_pos].strip()
        
        sections.append({
            "heading": heading,
            "text": content
        })
    
    return sections

# Example usage
pdf_path = "your_document.pdf"
sections = extract_sections_by_headings(pdf_path)

# Convert to JSON
json_output = json.dumps({
    "sections": sections
}, indent=4)

print(json_output)

# Optional: Save to file
# with open("output.json", "w") as f:
#     f.write(json_output)
