'''
Script to locally retrieve Zotero files
'''


import requests
import os

from pyzotero import zotero

zot = zotero.Zotero('10305233', 'user', 'rm5BW5WnXRfXlelUKKus1ZJP')
items = zot.top(limit=100)
# we've retrieved the latest five top-level items in our library
# we can print each item's item type and ID
for item in items:
    print('Item: %s | Key: %s' % (item['data']['itemType'], item['data']['key']))
    

download_dir = 'zotero_files'

# Ensure the download directory exists
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# Fetch all items from the Zotero library
items = zot.everything(zot.items())

# Download each attachment if it is a PDF
for item in items:
    # Check if the item has an attachment and is a PDF
    if 'data' in item and 'contentType' in item['data'] and item['data']['contentType'] == 'application/pdf':
        file_name = item['data'].get('filename', f"{item['data']['key']}.pdf")
        file_path = os.path.join(download_dir, file_name)
        
        # Extract the item key to use with zot.file()
        item_key = item['data']['key']
        print(f"Downloading {file_name}...")
        
        # Retrieve the file
        attachment = zot.file(item_key)
        
        if attachment:
            with open(file_path, 'wb') as f:
                f.write(attachment)
        else:
            print(f"Failed to download {file_name}.")
    else:
        print(f"Skipping item {item['data']['key']}: Not a PDF.")