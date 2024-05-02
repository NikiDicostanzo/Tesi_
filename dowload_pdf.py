import requests
from bs4 import BeautifulSoup
import os

# URL of the ACL Anthology page you want to start from
start_url = 'https://aclanthology.org/events/eacl-2023/#2023eacl-main' #'https://aclanthology.org/'


def find_pdf_links(url): #venues ->
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
   # print(soup.find_all('a'))
    #pdf_links = [link['href'] for link in soup.find_all('a') if link['href'].endswith('.pdf')]
    # Find all hyperlinks present on the webpage
    links = soup.find_all('a')
    pdf_links = []
    for link in links:
        # Check if the link points to a PDF file
        if 'pdf' in link.get('href', ''):#.endswith('.pdf'):
            # Construct the full URL for the PDF
         
            pdf_url = link.get("href")
            # Add the PDF URL to the list
            pdf_links.append(pdf_url)
    print(pdf_links)
    return pdf_links
    # Print the list of PDF links
   
def download_pdf(url, save_dir):
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(save_dir, url.split('/')[-1]), 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download {url}")


if __name__ == '__main__':
    # Example usage
    # Directory to save the PDFs
    save_dir = 'acl_anthology_pdfs_train/'
    os.makedirs(save_dir, exist_ok=True)
    print(len(os.listdir(save_dir)))
    path_json = os.listdir('HRDS/train/')
    list_name = [] # # devo prendere i pdf del TEST|
    start_link = 'https://aclanthology.org/'
    for j in path_json:
        name_pdf = '_'.join(j.split('_')[1:])
       # print(name_pdf)
       # print(start_link + name_pdf.replace('.json', '.pdf'))
        link = start_link + name_pdf.replace('.json', '.pdf')
        download_pdf(link, save_dir)
       # list_name.append()
        
    # #pdf_links = find_pdf_links(start_url)
    # #print(pdf_links)
    # for link in pdf_links:
    #     print(link)
    #     download_pdf(link, save_dir)

