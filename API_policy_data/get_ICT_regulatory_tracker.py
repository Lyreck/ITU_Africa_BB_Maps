import requests
import json
import csv


def call_api_ICT(countries_list=[],regions_list=[], year=2024):
    """
    Call the ITU API
    """

    countries = json.dumps(countries_list)

    if len(countries_list) > 0:
        url = f"https://www.itu.int/net4/itu-d/metrics/api/v1.1/tracker/entries?countries={countries}&brief=false&year={year}"
    elif len(regions_list) > 0:
        regions = json.dumps(regions_list)
        url = f"https://www.itu.int/net4/itu-d/metrics/api/v1.1/tracker/entries?regions={regions}&brief=false&year={year}"

    response = requests.get(url)
    data = response.json()

    return data

def filter_data(data, indicators_regulatory_authority, indicators_regulatory_mandate, indicators_regulatory_regime, indicators_competition_framework):
    """
    From the data returned by the API, filter only the indicators that we asked for
    """

    filtered_data = []

    for entry in data:
        country_dict = {k: entry[k] for k in ["year","region","country","regulatoryAuthorityScore","regulatoryMandateScore","regulatoryRegimeScore","competitionFrameworkScore","overallScore","generation"]}
        
        # Adding the different indicators we want, pillar per pillar.
        for d in entry["regulatoryAuthority"]:
            if d['area'] in indicators_regulatory_authority:
                country_dict[d['area']] = {"value": d.get("value"), "description": d.get("description")}

        for d in entry["regulatoryMandate"]:
            if d['area'] in indicators_regulatory_mandate:
                country_dict[d['area']] = {"value": d.get("value"), "description": d.get("description")}

        for d in entry["regulatoryRegime"]:
            if d['area'] in indicators_regulatory_regime:
                country_dict[d['area']] = {"value": d.get("value"), "description": d.get("description")}

        for d in entry["competitionFramework"]:
            if d['area'] in indicators_competition_framework:
                country_dict[d['area']] = {"value": d.get("value"), "description": d.get("description")}

        filtered_data.append(country_dict)
        
            
    return filtered_data

def write_to_csv(filtered_data, indicator_names, filename):
    """
    Write the filtered data to a CSV file
    """

    # Prepare CSV columns: base fields + indicator values + indicator descriptions
    base_fields = ["year", "region", "country", "regulatoryAuthorityScore", "regulatoryMandateScore", "regulatoryRegimeScore", "competitionFrameworkScore", "overallScore", "generation"]
    value_fields = [f"{ind}_value" for ind in indicator_names]
    desc_fields = [f"{ind}_desc" for ind in indicator_names]
    fieldnames = base_fields + value_fields + desc_fields

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in filtered_data:
            out = {k: row.get(k, "") for k in base_fields}
            for ind in indicator_names:
                val = row.get(ind, {})
                out[f"{ind}_value"] = val.get("value", "") if isinstance(val, dict) else ""
                out[f"{ind}_desc"] = val.get("description", "") if isinstance(val, dict) else ""
            writer.writerow(out)

def create_ICTRT_datasets():
    ################################################## These parameters can be changed ###################################################
    #change this list to get other countries
    countries_list = ["Côte d'Ivoire", "Benin", "Nigeria","Ethiopia","Kenya","Uganda","Burundi","Zambia","Malawi","Zimbabwe","Botswana"]

    # One list per pillar. There are 4 pillars in the ICT regulatory tracker.
    indicators_regulatory_authority = ["Separate telecom/ICT regulator",
                                       "Autonomy in decision making",
                                       "Accountability",
                                       "Enforcement power",
                                       "Sanctions or penalties imposed by regulator"]

    indicators_regulatory_mandate = ["Traditional mandate: entity in charge of quality of service obligations measures and service quality monitoring",
        "Entity in charge of universal service/access",
        "Consumer issues: entity responsible for comparative tariff information, consumer education and handling consumer complaints",
        "Operators required to publish Reference Interconnection Offer (RIO)"]

    indicators_regulatory_regime = ["Quality of service monitoring required",
        "Infrastructure sharing for mobile operators permitted",
        "Infrastructure sharing mandated",
        "Co-location/site sharing mandated",
        "Unbundled access to the local loop required", #broadband cost reduction directive (EU legislation)
        "National plan that involves broadband"]

    indicators_competition_framework = ["Level of competition in IMT (3G, 4G, etc.) services",
        "Level of competition in cable modem, DSL, fixed wireless broadband"]

    # List of all indicator names for the csv columns.
    indicator_names = [
        "Separate telecom/ICT regulator",
        "Autonomy in decision making",
        "Accountability",
        "Enforcement power",
        "Sanctions or penalties imposed by regulator",
        "Traditional mandate: entity in charge of quality of service obligations measures and service quality monitoring",
        "Entity in charge of universal service/access",
        "Consumer issues: entity responsible for comparative tariff information, consumer education and handling consumer complaints",
        "Operators required to publish Reference Interconnection Offer (RIO)",
        "Quality of service monitoring required",
        "Infrastructure sharing for mobile operators permitted",
        "Infrastructure sharing mandated",
        "Co-location/site sharing mandated",
        "Unbundled access to the local loop required", #broadband cost reduction directive (EU legislation)
        "National plan that involves broadband",
        "Level of competition in IMT (3G, 4G, etc.) services",
        "Level of competition in cable modem, DSL, fixed wireless broadband"] 
    
    regions_list = ["Europe"]
    year= 2020
    ################################################## These parameters can be changed ###################################################
    
    for year in [2020, 2022]:
        # Create dataset for countries in the list (11 Partner African Countries)
        # Call the API
        data = call_api_ICT(countries_list=countries_list, year=year)
        # Filter the data
        filtered_data = filter_data(data, indicators_regulatory_authority, indicators_regulatory_mandate, indicators_regulatory_regime, indicators_competition_framework)
        # Write the filtered data
        write_to_csv(filtered_data, indicator_names, filename=f'data/ICT_regulatory_tracker_{year}.csv')

        # Create dataset for Europe region.
        data = call_api_ICT(regions_list=regions_list, year=year)
        filtered_data = filter_data(data, indicators_regulatory_authority, indicators_regulatory_mandate, indicators_regulatory_regime, indicators_competition_framework)
        write_to_csv(filtered_data, indicator_names, filename=f'data/ICT_regulatory_tracker_Europe_{year}.csv')

if __name__ == "__main__":

    create_ICTRT_datasets()
    