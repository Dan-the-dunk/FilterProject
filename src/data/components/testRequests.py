import zipfile
import requests
from tqdm import tqdm



def download_data(url):

    url = "https://databank.worldbank.org/data/download/WDI_CSV.zip"
    response = requests.get(url, stream=True)
    with open("data/WDI_CSV.zip", mode="wb") as file:
        dl = 0
        for chunk in tqdm(response.iter_content(chunk_size=1024)): 
            if chunk:
                file.write(chunk)
                file.flush()
                
        
def unzip_data():

    with zipfile.ZipFile('data\WDI_CSV.zip', 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc='Extracting '):
            try:
                zip_ref.extract(member, 'data')
            except zipfile.error as e:
                pass