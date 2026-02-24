# two plots
# one for time, for each seq len
# one for throughput, for each seq len
# assume 512, 1024, 2048, 4096, 16384
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_grouped_bars(df, kernels, title, ylabel, output_name, value_style="number"):
    # Keep non-flash bars on the left and flash bars on the right.
    kernels = [k for k in kernels if "flash" not in k] + [k for k in kernels if "flash" in k]
    df_plot = df[df["Kernel Name"].isin(kernels)].copy()
    seq_lens = sorted(df_plot["seq_len"].unique())
    pivot = (
        df_plot.pivot_table(
            index="seq_len",
            columns="Kernel Name",
            values="Metric Value",
            aggfunc="first",
        )
        .reindex(seq_lens)
        .fillna(0.0)
    )

    x = np.arange(len(seq_lens))
    # Keep bar thickness consistent across plots; match the 3-bar plot thickness.
    bar_width = 0.5 / 3

    fig, ax = plt.subplots(figsize=(12, 6))
    all_bars = []
    for i, kernel in enumerate(kernels):
        offset = (i - (len(kernels) - 1) / 2) * bar_width
        bars = ax.bar(x + offset, pivot[kernel].values, bar_width, label=kernel)
        all_bars.append(bars)

    def format_value(v):
        if value_style == "percent":
            return f"{v:.0f}%"
        if value_style == "speedup":
            return f"{v:.2f}x"
        return f"{v:.2f}"

    max_y = max([float(np.max(b.datavalues)) for b in all_bars] + [0.0])
    label_offset = max(0.003 * max_y, 0.005)
    for bars in all_bars:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + label_offset,
                format_value(h),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Seq Len")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(seq_lens)
    ax.legend()
    ax.margins(y=0.12)
    plt.tight_layout()
    plt.savefig(output_name, dpi=300, bbox_inches=None)


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


def get_metric(df, kernel_name, metric_name, seq_len):
    return df.loc[
        (df["Kernel Name"] == kernel_name) & (df["Metric Name"] == metric_name) & (df["seq_len"] == seq_len),
        "Metric Value"
    ].iloc[0]


def clean_kernel_names(kernel_name):
    if 'flash_fwd_kernel' in kernel_name:
        return "flash_fwd"
    elif 'flash_bwd_dot_do_o_kernel' in kernel_name:
        return "flash_bwd_dot_do_o"
    elif 'flash_bwd_dq_kernel' in kernel_name:
        return "flash_bwd_dq"
    elif 'flash_bwd_dk_dv_kernel' in kernel_name:
        return "flash_bwd_dk_dv"
    elif "PyTorchMemEffAttention::AttentionKernel" in kernel_name:
        return "pytorch_fwd"
    elif "PyTorchMemEffAttention::AttentionBackwardKernel" in kernel_name:
        return "pytorch_bwd"
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

HDIMS = [64, 128]
SEQLENS  = [500, 512, 1000, 1024, 2000, 2048, 4000, 4096, 8000, 8192, 16000, 16384]
IS_CAUSALS = ["False", "True"]

for hdim in HDIMS:
    for is_causal in IS_CAUSALS:
        df_list = []

        for seqlen in SEQLENS:
            csv_file_path = f"{seqlen}_{hdim}_{is_causal}.csv"
            if not os.path.exists(csv_file_path):
                continue

            skip_rows = count_skip_rows(csv_file_path)
            df = pd.read_csv(csv_file_path, skiprows=skip_rows)

            filtered_df = filter_df(df)
            filtered_df["Kernel Name"] = filtered_df["Kernel Name"].apply(clean_kernel_names)
            filtered_df["Metric Name"] = filtered_df["Metric Name"].apply(clean_metric_names)
            filtered_df["seq_len"] = seqlen
            df_list.append(filtered_df)

        if not df_list:
            continue

        df_combined = pd.concat(df_list, axis=0, ignore_index=True)

        df_combined["Metric Value"] = (
            df_combined["Metric Value"].astype(str).str.replace(",", "", regex=False).astype(float)
        )

        df_duration = df_combined[df_combined["Metric Name"] == "duration"].copy()
        df_throughput = df_combined[df_combined["Metric Name"] == "compute throughput"].copy()
        causal_suffix = is_causal.lower()

        # Convert duration ns -> ms for easier comparison on plots
        df_duration["Metric Value"] = df_duration["Metric Value"] / 1_000_000

        def build_speedup_df(df, pytorch_kernel, flash_kernel, flash_label):
            pytorch = df[df["Kernel Name"] == pytorch_kernel][["seq_len", "Metric Value"]].copy()
            flash = df[df["Kernel Name"] == flash_kernel][["seq_len", "Metric Value"]].copy()
            merged = pytorch.merge(flash, on="seq_len", suffixes=("_pytorch", "_flash"))
            speedup = merged[["seq_len"]].copy()

            pytorch_rows = speedup.copy()
            pytorch_rows["Kernel Name"] = pytorch_kernel
            pytorch_rows["Metric Value"] = 1.0

            flash_rows = speedup.copy()
            flash_rows["Kernel Name"] = flash_label
            flash_rows["Metric Value"] = merged["Metric Value_pytorch"] / merged["Metric Value_flash"]

            return pd.concat([pytorch_rows, flash_rows], ignore_index=True)

        # Forward speedup: pytorch fixed to 1, flash = pytorch/flash
        df_fwd_speedup = build_speedup_df(df_duration, "pytorch_fwd", "flash_fwd", "flash_fwd")
        plot_grouped_bars(
            df_fwd_speedup,
            ["pytorch_fwd", "flash_fwd"],
            f"fwd, hdim = {hdim}, causal = {is_causal}, speedup",
            "Speedup (x)",
            f"forward_{hdim}_{causal_suffix}_speedup.png",
            value_style="speedup",
        )

        # Forward throughput: flash_fwd vs pytorch_fwd
        plot_grouped_bars(
            df_throughput,
            ["flash_fwd", "pytorch_fwd"],
            f"fwd, hdim = {hdim}, causal = {is_causal}, compute throughput",
            "Compute Throughput (%)",
            f"forward_{hdim}_{causal_suffix}_throughput.png",
            value_style="percent",
        )

        # Backward duration: sum flash_bwd_* and compare with pytorch_bwd
        flash_bwd_duration = (
            df_duration[df_duration["Kernel Name"].isin(["flash_bwd_dot_do_o", "flash_bwd_dq", "flash_bwd_dk_dv"])]
            .groupby("seq_len", as_index=False)["Metric Value"]
            .sum()
        )
        flash_bwd_duration["Kernel Name"] = "flash_bwd"

        pytorch_bwd_duration = df_duration[df_duration["Kernel Name"] == "pytorch_bwd"][
            ["seq_len", "Kernel Name", "Metric Value"]
        ].copy()

        df_bwd_duration = pd.concat([flash_bwd_duration, pytorch_bwd_duration], ignore_index=True)
        df_bwd_speedup = build_speedup_df(df_bwd_duration, "pytorch_bwd", "flash_bwd", "flash_bwd")
        plot_grouped_bars(
            df_bwd_speedup,
            ["pytorch_bwd", "flash_bwd"],
            f"bwd, hdim = {hdim}, causal = {is_causal}, speedup",
            "Speedup (x)",
            f"backward_{hdim}_{causal_suffix}_speedup.png",
            value_style="speedup",
        )

        # Backward throughput: pytorch_bwd vs flash_bwd_dq vs flash_bwd_dk_dv
        plot_grouped_bars(
            df_throughput,
            ["pytorch_bwd", "flash_bwd_dq", "flash_bwd_dk_dv"],
            f"bwd, hdim = {hdim}, causal = {is_causal}, compute throughput",
            "Compute Throughput (%)",
            f"backward_{hdim}_{causal_suffix}_throughput.png",
            value_style="percent",
        )
