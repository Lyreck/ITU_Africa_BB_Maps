import pandas as pd


if __name__ == "__main__":
    data = pd.read_csv("data/software.csv", index_col="Country")
    print(data)

    # Data for the bar chart of number of software used. I want a list of column names, and a list with the corresponding values.
    column_names = data.columns.tolist()

    numbers = data.agg('sum', axis="index").tolist()

    print(column_names)
    print(numbers)

    print(len(column_names)==len(numbers))


    # Data for Pie chart Open source. I need data in this format: { value: 1048, name: 'Search Engine' }

    closed_source = ["ESRI ArcGIS", "Stratix", "Carto", "Mapy", "Google Maps", "Wigeogis", "Salesforce"]
    open_source= 0
    for name, value in zip(column_names, numbers):
        if name not in closed_source: open_source += value

    print(f"{{ value: {numbers[-1]}, name: 'Countries using mainly Open Source' }}, \n {{ value: {len(data.index) - numbers[-1]}, name: 'Countries using mainly Closed Source' }}") #numbers[-1] contains the number of countries where "Open Source" == 1.
