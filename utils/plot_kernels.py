# two plots
# one for time, for each seq len
# one for throughput, for each seq len
# assume 512, 1024, 2048, 4096, 16384
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

def plot_grouped_bars(df, kernels, title, ylabel, output_name=None, value_style="number", ax=None):
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

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

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
    if own_fig:
        fig.tight_layout()
        if output_name:
            fig.savefig(output_name, dpi=300, bbox_inches=None)
        plt.close(fig)


def save_2x2_grid(
    hdim,
    direction,
    data_by_metric,
    metric_configs,
    output_name,
    is_causals,
):
    """Render a 2x2 grid combining causal/no-causal and metric variants."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"{direction}, hdim={hdim}", fontsize=16)
    has_data = False

    placement = [
        ("False", "speedup", (0, 0)),
        ("False", "throughput", (0, 1)),
        ("True", "speedup", (1, 0)),
        ("True", "throughput", (1, 1)),
    ]

    for is_causal, metric, (row, col) in placement:
        ax = axes[row, col]
        df = data_by_metric.get(metric, {}).get(is_causal)
        if df is None or df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
            ax.set_axis_off()
            continue

        config = metric_configs[metric]
        plot_grouped_bars(
            df,
            config["kernels"],
            f"causal={is_causal}, {config['title_suffix']}",
            config["ylabel"],
            value_style=config["value_style"],
            ax=ax,
        )
        has_data = True

    if has_data:
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(output_name, dpi=300, bbox_inches=None)
    plt.close(fig)


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

def read_profiler_csv(file_path):
    header_prefix = '"ID","Process ID","Process Name"'

    with open(file_path, "r") as file:
        lines = file.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith(header_prefix):
            header_idx = i
            break

    if header_idx is None:
        return pd.DataFrame(
            columns=["Kernel Name", "Metric Name", "Metric Unit", "Metric Value"]
        )

    csv_buffer = io.StringIO("".join(lines[header_idx:]))
    return pd.read_csv(csv_buffer)

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
    forward_data = {"speedup": {}, "throughput": {}}
    backward_data = {"speedup": {}, "throughput": {}}

    for is_causal in IS_CAUSALS:
        df_list = []

        for seqlen in SEQLENS:
            csv_file_path = f"{seqlen}_{hdim}_{is_causal}.csv"
            if not os.path.exists(csv_file_path):
                continue

            df = read_profiler_csv(csv_file_path)

            if df.empty:
                continue

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
        forward_data["speedup"][is_causal] = df_fwd_speedup

        # Forward throughput: flash_fwd vs pytorch_fwd
        forward_data["throughput"][is_causal] = df_throughput.copy()

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
        backward_data["speedup"][is_causal] = df_bwd_speedup

        # Backward throughput: pytorch_bwd vs flash_bwd_dq vs flash_bwd_dk_dv
        backward_data["throughput"][is_causal] = df_throughput.copy()

    forward_configs = {
        "speedup": {
            "kernels": ["pytorch_fwd", "flash_fwd"],
            "ylabel": "Speedup",
            "value_style": "speedup",
            "title_suffix": "speedup",
        },
        "throughput": {
            "kernels": ["flash_fwd", "pytorch_fwd"],
            "ylabel": "Compute Throughput (%)",
            "value_style": "percent",
            "title_suffix": "compute throughput",
        },
    }
    save_2x2_grid(
        hdim,
        "fwd",
        forward_data,
        forward_configs,
        f"forward_{hdim}_combined.png",
        IS_CAUSALS,
    )

    backward_configs = {
        "speedup": {
            "kernels": ["pytorch_bwd", "flash_bwd"],
            "ylabel": "Speedup",
            "value_style": "speedup",
            "title_suffix": "speedup",
        },
        "throughput": {
            "kernels": ["pytorch_bwd", "flash_bwd_dq", "flash_bwd_dk_dv"],
            "ylabel": "Compute Throughput (%)",
            "value_style": "percent",
            "title_suffix": "compute throughput",
        },
    }
    save_2x2_grid(
        hdim,
        "bwd",
        backward_data,
        backward_configs,
        f"backward_{hdim}_combined.png",
        IS_CAUSALS,
    )
