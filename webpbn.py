import requests
import xml.etree.ElementTree as ET
from io import StringIO


def fetch_webpbn(num):
    data = {
        "go": 1,
        "id": num,
        "xml_clue": "on",
        "fmt": "xml",
        "xml_soln": "on"}
    url = "https://webpbn.com/export.cgi/webpbn%06i.sgriddler" % num
    try:
        text = requests.post(url, data).text
        return parse_clues(text)
    except BaseException:
        return None


def parse_clues(xml_data):
    # Parse the XML data
    root = ET.parse(StringIO(xml_data)).getroot()

    # Check if colors are black and white only
    for color in root.findall(".//color"):
        if color.get('name') not in {'black', 'white'}:
            return None

    # Initialize formatted strings for rows and columns
    rows_formatted = ""
    columns_formatted = ""

    # Find and process clues
    for clue in root.findall(".//clues"):
        # Extract and format each line
        formatted_clue = ""
        for line in clue.findall('./line'):
            formatted_clue += ' '.join(count.text for count in line.findall('./count')) + '\n'

        # Add to rows or columns
        if clue.get('type') == 'columns':
            columns_formatted += formatted_clue
        elif clue.get('type') == 'rows':
            rows_formatted += formatted_clue

    # Concatenate rows and columns with separator
    return rows_formatted.strip() + "\n---\n" + columns_formatted.strip()
