import requests
import pandas as pd

def collect_data():
    # Example: Collect data from a hypothetical API
    url = "http://example.com/api/data"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data)
    df.to_csv('data/collected_data.csv', index=False)

if __name__ == "__main__":
    collect_data()