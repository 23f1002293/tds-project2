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
