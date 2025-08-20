import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patheffects
import textwrap
from adjustText import adjust_text
import numpy as np
import os


def process(data):
    # Set the first column as the index (e.g., 'Score')
    data.set_index(data.columns[0], inplace=True)


    # Order countries by ascending total score
    country_totals = data.sum(axis=0)
    sorted_countries = country_totals.sort_values(ascending=True).index
    data = data[sorted_countries]

    # Normalize data between 0 and 1 for better contrast, then apply a higher power
    data = (data - data.min().min()) / (data.max().max() - data.min().min())
    # data = data.applymap(lambda x: x**2)

    return data

def wrap_labels(labels, width):
        return ['\n'.join(textwrap.wrap(label, width)) for label in labels]

def draw_heatmap(data, show_scores=False, show_figures=True,transparent=True):

    # Create the heatmap
    plt.figure(figsize=(12, 8))
    if show_scores:
        plt.figtext(0.5, 0.01, "Maximum score is 109.96.", ha="center", fontsize=10, color="gray")
    else:
        plt.figtext(0.5, 0.01, "NB: data is normalized to have a maximum of 1.", ha="center", fontsize=10, color="gray")

    new_index = ['Broadband Policy and Planning', 'Infrastructure mapping',
        'Service Mapping and Regulatory Monitoring',
        'Data collection verification and accuracy',
        'Infrastructure coverage and resilience', 'Total']
    data=data.reindex(new_index) #make sure that Total is the last Row.

    heatmap = sns.heatmap(data, annot=show_scores, cmap='YlGnBu', linewidths=.5)

    # Add titles and labels
    # heatmap.set_title('Maturity Matrix Heatmap', fontsize=16)
    heatmap.set_xlabel('Countries', fontsize=12)
    heatmap.set_ylabel('Sections', fontsize=12)
    heatmap.set_yticklabels(wrap_labels(data.index, 15), fontsize=10)

    # Increase left and bottom margin to fit wrapped labels
    plt.subplots_adjust(left=0.2)
    plt.subplots_adjust(bottom=0.2)


    # Save the heatmap as an image
    plt.savefig(f'graphs/heatmap_ShowScores{show_scores}.png', transparent=transparent)

    # Display the heatmap
    if show_figures:
        plt.show()

def draw_country_heatmaps(data, transparent=True, show_figures=True):
    # Ensure output directory exists
    output_dir = "country_heatmaps"
    os.makedirs("graphs/"+output_dir, exist_ok=True)

    for country in data.columns:
        plt.figure(figsize=(8, 1.5))
        # Select only the current country (as a DataFrame)
        country_data = data[[country]].T.T  # Keeps index and column names
        #sns.heatmap(country_data, annot=False, cmap='YlGnBu', cbar=False, linewidths=.5)
        sns.heatmap(
            country_data,
            annot=True,
            cmap='tab10',  # Use a discrete/categorical palette
            cbar=False,
            linewidths=.5
        )
        plt.title(f"Heatmap for {country}", fontsize=12)
        plt.yticks(fontsize=10)
        plt.xticks([])
        plt.tight_layout()
        plt.savefig(f"graphs/{output_dir}/heatmap_{country}.png", transparent=transparent)
        plt.close()

def draw_country_bubble_lines(data, transparent=True, show_figures=True):
    # Ensure output directory exists
    output_dir = "country_bubble_lines"
    os.makedirs("graphs/"+output_dir, exist_ok=True)

    for country in data.columns:
        plt.figure(figsize=(10, 3))
        scores = data[country].values
        sections = list(data.index)
        x = np.arange(len(sections))
        # Normalize bubble sizes for better visibility
        sizes = (scores - np.min(scores)) / (np.ptp(scores) + 1e-8) * 800 + 100
    
        wrapped_sections = ['\n'.join(textwrap.wrap(sec, 18)) for sec in sections]

        plt.scatter(x, [0]*len(x), s=sizes, c=scores, cmap='YlGnBu', alpha=0.8, edgecolors='k')
        plt.yticks([])
        plt.xticks(x, wrapped_sections, rotation=45, ha='right', fontsize=9)
        # plt.title(f"Overview of the survey answers of {country} (bigger is more mature)", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"graphs/{output_dir}/bubble_line_{country}.png", transparent=transparent)
        if show_figures:
            plt.show()
        plt.close()


def draw_2D_plot(data, x_row="Infrastructure mapping", y_row="Service Mapping and Regulatory Monitoring", transparent=True, show_figures=True):
    """
    x_row and y_row can vary among: 
        - Total, 
        - Broadband Policy and Planning,
        - Infrastructure mapping,
        - Service Mapping and Regulatory Monitoring,
        - Data collection verification and accuracy,
        - Infrastructure coverage and resilience
    y_row can also be "Maturity Level".
    """


    x = data.loc[x_row]
    if y_row == "Maturity Level":
        map_dict = {"Nigeria": "Advanced", "Kenya":"Advanced", "Benin":"Advanced", "Côte d'Ivoire": "Advanced", "Malawi": "Medium" ,"Uganda": "Medium", "Zimbabwe": "Medium", "Zambia": "Medium", "Burundi": "Initial Stage", "Botswana":"Initial Stage","Ethiopia":"Initial Stage" }
        y = x.index.to_series().map(map_dict)
    else: 
        y = data.loc[y_row]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y)

    texts = [ax.text(x[country], y[country], country, #buggy but manages to reduce overlaps.
                     fontsize=10,
                    fontweight='bold',
                    ha='center',
                    va='center') for country in data.columns]
    adjust_text(texts)

    ax.set_xlabel(x_row)
    ax.set_ylabel(y_row)
    # title = f"Country Scores: {x_row} & {y_row}"
    # ax.set_title(textwrap.fill(title, width=50))
    ax.grid(True)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f'graphs/2d_plot_{x_row}_{y_row}.png', transparent=transparent)
    if show_figures:
        plt.show()

def draw_countries_maturity(data, x_row="Maturity Level", y_row="Total", draw_spline=False, transparent=True, show_figures=True):
    """
    Draws a scatter plot with three ITU Blue color zones for country maturity levels,
    with a legend for each zone, no grid, and no axis graduations.
    - put draw_spline to True if you want to interpolate the points with a spline curve.
    """

    # Define groupings
    advanced = ["Nigeria", "Kenya", "Benin", "Côte d'Ivoire"]
    intermediate = ["Malawi", "Uganda", "Zimbabwe", "Zambia"]
    initial = ["Burundi", "Botswana", "Ethiopia"]

    # ITU Blue shades (light to dark)
    blue_advanced = "#0033A0"      # ITU Blue
    blue_intermediate = "#4F81BD"  # Lighter blue
    blue_initial = "#B4C7E7"       # Very light blue

    y = data.loc[y_row]
    if y_row == "Total":
        y = (y - y.min()) / (y.max() - y.min())
    y.sort_values(ascending=True, inplace=True)

    if x_row == "Maturity Level":
        map_dict = {
            "Nigeria": 11, "Kenya": 9, "Benin": 10, "Côte d'Ivoire": 8,
            "Malawi": 7, "Uganda": 6, "Zimbabwe": 5, "Zambia": 4,
            "Burundi": 3, "Botswana": 2, "Ethiopia": 1
        }
        x = y.index.to_series().map(map_dict)
        x = (x - x.min()) / (x.max() - x.min())
        x.sort_values(ascending=True, inplace=True)
    else:
        x = data.loc[x_row].sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Compute x ranges for each group
    group_x = {
        "advanced": x[x.index.isin(advanced)],
        "intermediate": x[x.index.isin(intermediate)],
        "initial": x[x.index.isin(initial)]
    }
    adv_min, adv_max = group_x["advanced"].min(), group_x["advanced"].max()
    int_min, int_max = group_x["intermediate"].min(), group_x["intermediate"].max()
    ini_min, ini_max = group_x["initial"].min(), group_x["initial"].max()

    # Draw colored zones (axvspan) and create legend handles
    handles = []
    handles.append(ax.axvspan((int_max+adv_min)/2, adv_max+0.05, color=blue_advanced, alpha=0.12, zorder=0, label="Advanced"))
    handles.append(ax.axvspan((ini_max+int_min)/2, (int_max+adv_min)/2, color=blue_intermediate, alpha=0.12, zorder=0, label="Intermediate"))
    handles.append(ax.axvspan(ini_min-0.05, (ini_max+int_min)/2, color=blue_initial, alpha=0.12, zorder=0, label="Initial Stage"))

    # Scatter plot
    ax.scatter(x, y, color="#0033A0", zorder=2)

    if draw_spline:

        from scipy.interpolate import make_interp_spline
        x_array,y_array = np.sort(x.to_numpy()), np.sort(y.to_numpy())

        spl=make_interp_spline(x_array,y_array)

        x2_array=np.linspace(0,1,100)
        ax.plot(x2_array, spl(x2_array))

    # Annotate points
    for country in data.columns:
        if country in ['Burundi', "Malawi", "Uganda", 'Nigeria']:
            ax.annotate(
                country,
                (x[country], y[country]),
                fontsize=10,
                fontweight='bold',
                xytext=(5, 5),
                textcoords='offset points',
                ha='right',                 #These countries are at the extremity of the region, so we align the text such that the *right* of the text is next to the specified point.
                va='bottom',
                path_effects=[patheffects.withStroke(linewidth=3, foreground="white")]
            )
        else: 
            ax.annotate(
                country,
                (x[country], y[country]),
                fontsize=10,
                fontweight='bold',
                xytext=(5, 5),
                textcoords='offset points',
                ha='left',
                va='bottom',
                path_effects=[patheffects.withStroke(linewidth=3, foreground="white")]
            )

    # Remove grid and axis graduations
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add legend for color zones
    legend_labels = ["Advanced", "Intermediate", "Initial Stage"]
    legend_colors = [blue_advanced, blue_intermediate, blue_initial]
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, edgecolor='none', alpha=0.3, label=l) for c, l in zip(legend_colors, legend_labels)]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=12, frameon=True)

    ax.set_xlabel(x_row)
    ax.set_ylabel(y_row)
    # title = f"Country Scores: {x_row} & {y_row}"
    # ax.set_title(textwrap.fill(title, width=50))
    plt.tight_layout()
    plt.savefig(f'graphs/2d_curve_{x_row}_{y_row}.png', transparent=transparent)
    if show_figures:
        plt.show()
    plt.close()

def draw_curve_old(data, x_row="Maturity Level", y_row="Total", transparent=True, show_figures=True):
    """
    
    """

    # Define groupings
    advanced = ["Nigeria", "Kenya", "Benin", "Côte d'Ivoire"]
    intermediate = ["Malawi", "Uganda", "Zimbabwe", "Zambia"]
    initial = ["Burundi", "Botswana", "Ethiopia"]

    # ITU Blue shades (light to dark)
    blue_advanced = "#0033A0"      # ITU Blue
    blue_intermediate = "#4F81BD"  # Lighter blue
    blue_initial = "#B4C7E7"       # Very light blue

    y = data.loc[y_row]
    if y_row == "Total":
        y = (y - y.min()) / (y.max() - y.min())

    if x_row == "Maturity Level":
        map_dict = {
            "Nigeria": 1, "Kenya": 3, "Benin": 2, "Côte d'Ivoire": 4,
            "Malawi": 5, "Uganda": 6, "Zimbabwe": 7, "Zambia": 8,
            "Burundi": 9, "Botswana": 10, "Ethiopia": 11
        }
        x = y.index.to_series().map(map_dict)
        x = (x - x.min()) / (x.max() - x.min())
    else:
        x = data.loc[x_row]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Compute x ranges for each group
    x_vals = x.copy()
    group_x = {
        "advanced": x[x.index.isin(advanced)],
        "intermediate": x[x.index.isin(intermediate)],
        "initial": x[x.index.isin(initial)]
    }
    # Get min/max for each group
    adv_min, adv_max = group_x["advanced"].min(), group_x["advanced"].max()
    int_min, int_max = group_x["intermediate"].min(), group_x["intermediate"].max()
    ini_min, ini_max = group_x["initial"].min(), group_x["initial"].max()

    # Draw colored zones (axvspan)
    ax.axvspan(adv_min, adv_max, color=blue_advanced, alpha=0.12, zorder=0)
    ax.axvspan(int_min, int_max, color=blue_intermediate, alpha=0.12, zorder=0)
    ax.axvspan(ini_min, ini_max, color=blue_initial, alpha=0.12, zorder=0)

    # Scatter plot
    ax.scatter(x, y, color="#0033A0", zorder=2)


    texts = [ax.text(x[country], y[country], country, #buggy but manages to reduce overlaps.
                     fontsize=10,
                    fontweight='bold',
                    ha='center',
                    va='center') for country in data.columns]
    adjust_text(texts)

    ax.set_xlabel(x_row)
    ax.set_ylabel(y_row)
    title = f"Country Scores: {x_row} & {y_row}"
    ax.set_title(textwrap.fill(title, width=50))
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f'graphs/2d_curve_{x_row}_{y_row}.png', transparent=transparent)
    if show_figures:
        plt.show()


## Make a lot of graphs to overwhelm
def draw_bubble_plot(df, transparent=True, show_figures=True):
    # Reset index to turn the row labels into a column
    df_reset = df.reset_index().rename(columns={df.index.name or df_reset.columns[0]: "Section"})
    df_melted = df_reset.melt(id_vars=["Section"], var_name="Country", value_name="Value")

    # Normalize values for bubble sizes
    max_val = df_melted["Value"].max()
    df_melted["BubbleSize"] = df_melted["Value"] / max_val * 3000  # scale for visibility

    # Define unique sections and angles
    categories = df_reset["Section"].unique()
    n_categories = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False)
    angle_map = dict(zip(categories, angles))
    df_melted["Angle"] = df_melted["Section"].map(angle_map)

    # Set radial position for each country
    country_list = df_melted["Country"].unique()
    radius_map = dict(zip(country_list, np.linspace(1, len(country_list), len(country_list))))
    df_melted["Radius"] = df_melted["Country"].map(radius_map)

    # Assign a unique color to each country
    palette = sns.color_palette("tab10", len(country_list))
    color_map = dict(zip(country_list, palette))

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw={'polar': True})

    # Plot each country with its color
    for country in country_list:
        country_data = df_melted[df_melted["Country"] == country]
        ax.scatter(
            country_data["Angle"],
            country_data["Radius"],
            s=country_data["BubbleSize"],
            alpha=0.7,
            label=country,
            edgecolors='white',
            color=color_map[country]
        )

    wrapped_categories = ['\n'.join(textwrap.wrap(cat, 18)) for cat in categories]

    # Set the axes
    ax.set_xticks(angles)
    ax.set_xticklabels(wrapped_categories, fontsize=10)
    ax.set_yticklabels([])
    ax.tick_params(axis='x', pad=30)
    ax.tick_params(axis='y', pad=20)
    # ax.set_title("Broadband Mapping Maturity, by country and area.", fontsize=16)

    # Unique legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc='upper right',
        bbox_to_anchor=(1.05, 0, 0.25, 1),  # (x, y, width, height) in axes fraction
        fontsize=14,
        markerscale=0.5,
        borderaxespad=0,
        mode='expand'
    )

    # Example: annotate only the largest bubble for each section
    for section in categories:
        section_data = df_melted[df_melted["Section"] == section]
        max_row = section_data.loc[section_data["Value"].idxmax()]
        ax.text(
            max_row["Angle"], max_row["Radius"], max_row["Country"],
            fontsize=8, color='black', ha='center', va='center', alpha=0.7
        )

    plt.tight_layout()
    plt.savefig('graphs/circular_bubbles.png', transparent=transparent)
    if show_figures:
        plt.show()






if __name__ == "__main__":
    # Load the CSV data with ISO-8859-1 encoding to handle special characters
    data = pd.read_csv('data/clean_data_heatmap.csv', encoding='UTF-8')
    os.makedirs("graphs", exist_ok=True)

    processed_data = process(data)

    # draw_heatmap(processed_data, transparent=False, show_figures=False)
    # draw_heatmap(data, show_scores=True, show_figures=False,transparent=False) #for internal use only.

    # draw_2D_plot(processed_data,transparent=False, show_figures=False)
    # draw_2D_plot(processed_data,x_row="Data collection verification and accuracy", y_row="Infrastructure coverage and resilience", transparent=False, show_figures=False)
    # draw_2D_plot(processed_data,x_row="Broadband Policy and Planning", y_row="Service Mapping and Regulatory Monitoring", transparent=False, show_figures=False)
    # draw_2D_plot(processed_data,x_row="Total", y_row="Maturity Level",transparent=False, show_figures=False) #Graph asked for by Elind


    # draw_country_heatmaps(processed_data, transparent=False, show_figures=False)
    # draw_country_bubble_lines(processed_data, transparent=False, show_figures=False)

    # draw_bubble_plot(data, transparent=False, show_figures=False)

    # draw_countries_maturity(data, x_row="Maturity Level", y_row="Total", draw_spline=False, transparent=False, show_figures=True)