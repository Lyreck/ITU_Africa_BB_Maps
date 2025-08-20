import requests
import json
import csv


def call_api_G5(year=2023):
    """
    Call the ITU API
    """

    url = f"https://www.itu.int/net4/itu-d/metrics/api/v1.1/benchmark/entries?year={year}"

    response = requests.get(url)
    data = response.json()

    return data

def create_G5_datasets():
    ################################################## These parameters can be changed ###################################################
    #change this list to get other countries
    countries_list = ["Côte d'Ivoire", "Benin", "Nigeria","Ethiopia","Kenya","Uganda","Burundi","Zambia","Malawi","Zimbabwe","Botswana"]
    year = 2023
    ################################################## These parameters can be changed ###################################################

    # Call the API
    data = call_api_G5(year=year)

    with open(f'data/G5_benchmark_{year}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    create_G5_datasets()