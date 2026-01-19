import re

def clean_pdf_text(text: str) -> str:
    """
    Clean PDF text by removing extra spaces between characters
    and normalizing whitespace
    """
    # Fix spaced characters (e.g., "S k i l l s" -> "Skills")
    text = re.sub(r'(?<=\w)\s+(?=\w)', '', text)
    
    # Normalize multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common PDF extraction issues
    text = text.replace('\n\n\n', '\n\n')
    text = text.replace('  ', ' ')
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

