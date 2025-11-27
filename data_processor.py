import csv
import io
import re

def sum_numbers_in_csv(csv_content: str) -> float:
    """Reads CSV content and returns the sum of all numbers found."""
    total_sum = 0.0
    f = io.StringIO(csv_content)
    reader = csv.reader(f)
    for row in reader:
        for item in row:
            try:
                total_sum += float(item)
            except ValueError:
                pass  # Ignore non-numeric values
    return total_sum

def extract_secret_code(text_content: str) -> str:
    """Extracts a secret code from text content using a general-purpose regex."""
    # This regex looks for a standalone alphanumeric string of 16+ characters
    # that is not part of a URL, which is a common pattern for secrets.
    match = re.search(r'(?<!\w)[a-zA-Z0-9]{16,}(?!\w)', text_content)
    if match:
        return match.group(0) # Return the entire matched string
    
    # Fallback for HTML comments
    match = re.search(r'<!--\s*([a-zA-Z0-9]{16,})\s*-->', text_content)
    if match:
        return match.group(1)
        
    return ""
