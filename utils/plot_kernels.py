# two plots
# one for time, for each seq len
# one for throughput, for each seq len
# assume 512, 1024, 2048, 4096, 16384
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_graph(df, metric_value, metric_unit):
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # We need to separate the data for each seq_len and kernel type
    df_flash = df[df['Kernel Name'] == 'flash attention']
    df_memory_efficient = df[df['Kernel Name'] == 'memory efficient attention']

    # Set the width of the bars
    bar_width = 0.2

    # Set the positions for the bars (grouped next to each other)
    index = np.arange(len(df_flash))  # or len(df_memory_efficient), they should be the same length

    # Create the bars for "flash attention" and "memory efficient attention"
    bars_memory = ax.bar(index, df_memory_efficient[metric_value], bar_width, label='Memory Efficient Attention', color='#2ca02c')
    bars_flash = ax.bar(index + bar_width, df_flash[metric_value], bar_width, label='Flash Attention', color='#9467bd')

    # Add labels, title, and adjust x-ticks to match seq_len
    ax.set_xlabel('Seq Len')
    ax.set_ylabel(metric_value)
    ax.set_title(f'{metric_value} Comparison for Flash vs Memory Efficient Attention')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(df_flash['seq_len'])
    ax.legend()

    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}{metric_unit}',
                    ha='center', va='bottom', fontsize=10)#, fontweight='bold')

    add_labels(bars_memory)
    add_labels(bars_flash)


    # Show the plot
    plt.xticks(rotation=45)
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"{metric_value.replace(' ', '_')}.png", dpi=300, bbox_inches=None)

    return 0


# remove pytorch reduction kernels
# remove unwanted columns
def filter_df(df):
    # remove unwanted kernels
    keywords = ['void at::']

    pattern = '|'.join(keywords)

    df_filtered = df[~df['Kernel Name'].str.contains(pattern, case=False)]

    # remove unwanted columns
    columns_to_keep = ["Kernel Name", "Metric Name", "Metric Unit", "Metric Value"]
    df_filtered = df_filtered[columns_to_keep].copy()


    return df_filtered


def clean_kernel_names(kernel_name):
    if 'flash' in kernel_name:
        return "flash attention"
    elif "fmha" in kernel_name:
        return "memory efficient attention"
    else:
        return kernel_name

def clean_metric_names(metric_name):
    if metric_name == "gpu__time_duration.sum":
        return "duration"
    elif metric_name == "sm__throughput.avg.pct_of_peak_sustained_elapsed":
        return "compute throughput"
    else:
        return metric_name

def count_skip_rows(file_path):
    marker_line = "==PROF== Disconnected from process"
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if marker_line in line:
                return i + 1  # Skip the marker line as well
    return 0  # Default to 0 if the marker is not found

def compute_speed_up(df):
    df["Speed Up"] = 1

    for index, row in df.iterrows():
        if row["Kernel Name"] == "flash attention":
            # Get the corresponding "memory efficient attention" value for the same seq_len
            corresponding_memory_efficient_row = df[(df["Kernel Name"] == "memory efficient attention") &
                                                    (df["seq_len"] == row["seq_len"])]


            df.at[index, "Speed Up"] = corresponding_memory_efficient_row["Metric Value"].values[0] / row["Metric Value"]

    return df

df_list = []
NUMS  = [512, 1024, 2048, 4096, 8192, 16384]

for num in NUMS:
    # Path to your CSV file
    csv_file_path = f"{num}.csv"
    skip_rows = count_skip_rows(csv_file_path)
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path, skiprows=skip_rows)

    #columns_to_keep = ["Kernel Name", "Metric Name", "Metric Unit", "Metric Value"]
    # = df[columns_to_keep].copy()
    filtered_df = filter_df(df)

    filtered_df["Kernel Name"] = filtered_df["Kernel Name"].apply(clean_kernel_names)
    filtered_df["Metric Name"] = filtered_df["Metric Name"].apply(clean_metric_names)
    filtered_df["seq_len"] = num
    #print(filtered_df.head()
    df_list.append(filtered_df)


df_combined = pd.concat(df_list, axis=0, ignore_index=True)

#print(df_combined)

df_duration = df_combined[df_combined['Metric Name'] == "duration"].copy()
df_duration["Metric Value"] = df_duration["Metric Value"].replace({',': ''}, regex=True).astype(float) / 1_000_000
df_duration = compute_speed_up(df_duration)
df_compute_throughput = df_combined[df_combined['Metric Name'] == "compute throughput"].copy()
df_compute_throughput["Compute Throughput"] = df_compute_throughput["Metric Value"].astype(float)
print(df_duration)
print(df_compute_throughput)

plot_graph(df_duration, "Speed Up", "x")
plot_graph(df_compute_throughput, "Compute Throughput", "%")