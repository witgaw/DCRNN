from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os

import numpy as np
import pandas as pd


def copy_sensor_graph_data(dataset_name, output_dir):
    """Copy relevant sensor graph files to the output directory."""
    sensor_graph_dir = "data/sensor_graph"
    output_sensor_dir = os.path.join(output_dir, "sensor_graph")

    # Create sensor_graph subdirectory
    os.makedirs(output_sensor_dir, exist_ok=True)

    # Mapping of datasets to their sensor files
    sensor_files = {
        "METR-LA": [
            "graph_sensor_locations.csv",
            "distances_la_2012.csv",
            "adj_mx.pkl",  # 207x207 adjacency matrix
        ],
        "PEMS-BAY": [
            "graph_sensor_locations_bay.csv",
            "distances_bay_2017.csv",
            "adj_mx_bay.pkl",  # 325x325 adjacency matrix
        ],
    }

    if dataset_name in sensor_files:
        for filename in sensor_files[dataset_name]:
            src_path = os.path.join(sensor_graph_dir, filename)
            dst_path = os.path.join(output_sensor_dir, filename)

            if os.path.exists(src_path):
                import shutil

                if filename.endswith(".pkl"):
                    # Convert pickle adjacency matrix to NPY format
                    try:
                        import pickle

                        with open(src_path, "rb") as f:
                            data = pickle.load(f, encoding="latin1")
                        sensor_ids, sensor_id_to_ind, adj_mx = data

                        # Save as NPY file instead
                        npy_filename = filename.replace(".pkl", ".npy")
                        npy_path = os.path.join(output_sensor_dir, npy_filename)
                        np.save(npy_path, adj_mx)
                        print(
                            f"  Converted {filename} to {npy_filename} (shape: {adj_mx.shape})"
                        )

                        # Also save sensor mapping as JSON for reference
                        json_filename = filename.replace(".pkl", "_mapping.json")
                        json_path = os.path.join(output_sensor_dir, json_filename)
                        with open(json_path, "w") as f:
                            import json

                            json.dump(
                                {
                                    "sensor_ids": sensor_ids,
                                    "description": f"Adjacency matrix for {dataset_name}",
                                    "shape": list(adj_mx.shape),
                                    "non_zero_entries": int(np.sum(adj_mx > 0)),
                                    "sparsity": float(
                                        np.sum(adj_mx == 0) / adj_mx.size
                                    ),
                                    "parameters": {
                                        "normalized_k": 0.1,
                                        "method": "gaussian_kernel_exp(-d²/σ²)",
                                        "distance_threshold": "values < 0.1 set to 0",
                                    },
                                },
                                f,
                                indent=2,
                            )
                        print(f"  Created {json_filename} with metadata")
                    except Exception as e:
                        print(f"  Warning: Could not convert {filename}: {e}")
                        # Fall back to copying the original file
                        shutil.copy2(src_path, dst_path)
                        print(f"  Copied {filename} (pickle format)")
                else:
                    shutil.copy2(src_path, dst_path)
                    print(f"  Copied {filename} to sensor_graph/")
            else:
                print(f"  Warning: {filename} not found")

    # Also create a README for the sensor graph data
    readme_content = f"""# Sensor Graph Data for {dataset_name}

This directory contains spatial information about the traffic sensors used in the {dataset_name} dataset.

## Files

"""

    if dataset_name == "METR-LA":
        readme_content += """- `graph_sensor_locations.csv`: Sensor coordinates (latitude, longitude) for 207 sensors
- `distances_la_2012.csv`: Pairwise distances between sensors in meters
- `adj_mx.npy`: Pre-computed adjacency matrix (207×207) for graph neural networks
- `adj_mx_mapping.json`: Metadata and parameters used to generate the adjacency matrix

## Usage

```python
import pandas as pd
import numpy as np

# Load sensor locations
locations = pd.read_csv('sensor_graph/graph_sensor_locations.csv')
print(f"Dataset has {len(locations)} sensors")

# Load distances (for custom graph construction)
distances = pd.read_csv('sensor_graph/distances_la_2012.csv')

# Load pre-computed adjacency matrix
adj_matrix = np.load('sensor_graph/adj_mx.npy')
print(f"Adjacency matrix shape: {adj_matrix.shape}")
```
"""
    elif dataset_name == "PEMS-BAY":
        readme_content += """- `graph_sensor_locations_bay.csv`: Sensor coordinates (latitude, longitude) for 325 sensors  
- `distances_bay_2017.csv`: Pairwise distances between sensors in meters
- `adj_mx_bay.npy`: Pre-computed adjacency matrix (325×325) for graph neural networks
- `adj_mx_bay_mapping.json`: Metadata and parameters used to generate the adjacency matrix

## Usage

```python
import pandas as pd
import numpy as np

# Load sensor locations
locations = pd.read_csv('sensor_graph/graph_sensor_locations_bay.csv')
print(f"Dataset has {len(locations)} sensors")

# Load distances (for custom graph construction)
distances = pd.read_csv('sensor_graph/distances_bay_2017.csv')

# Load pre-computed adjacency matrix
adj_matrix = np.load('sensor_graph/adj_mx_bay.npy')
print(f"Adjacency matrix shape: {adj_matrix.shape}")
```
"""

    readme_content += """
## Coordinate System

- Coordinates are in WGS84 (latitude, longitude)
- Distances are in meters
- Use this data to construct the adjacency matrix for graph neural networks

## Citation

This spatial data is part of the original dataset used in:

> Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018). Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting. ICLR 2018.
"""

    with open(os.path.join(output_sensor_dir, "README.md"), "w") as f:
        f.write(readme_content)

    print("  Created sensor_graph/README.md")


def arrays_to_dataframe_batched(x, y, x_offsets, y_offsets, batch_size=1000):
    """
    Convert multi-dimensional arrays to DataFrame format in batches to save memory.

    Args:
        x: Input features array (num_samples, input_length, num_nodes, input_dim)
        y: Target values array (num_samples, output_length, num_nodes, output_dim)
        x_offsets: Input time offsets
        y_offsets: Output time offsets
        batch_size: Number of samples to process at once

    Returns:
        pandas.DataFrame: Flattened data suitable for Parquet
    """
    num_samples, input_length, num_nodes, input_dim = x.shape
    _, output_length, _, output_dim = y.shape

    all_dfs = []

    # Process in batches to avoid memory issues
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        print(f"  Processing samples {batch_start:,} to {batch_end:,}...")

        records = []

        for sample_idx in range(batch_start, batch_end):
            for node_idx in range(num_nodes):
                record = {
                    "node_id": node_idx,
                }

                # Add input features
                for time_idx in range(input_length):
                    for dim_idx in range(input_dim):
                        offset = x_offsets[time_idx]
                        col_name = f"x_t{offset:+d}_d{dim_idx}"
                        record[col_name] = float(
                            x[sample_idx, time_idx, node_idx, dim_idx]
                        )

                # Add target values
                for time_idx in range(output_length):
                    for dim_idx in range(output_dim):
                        offset = y_offsets[time_idx]
                        col_name = f"y_t{offset:+d}_d{dim_idx}"
                        record[col_name] = float(
                            y[sample_idx, time_idx, node_idx, dim_idx]
                        )

                records.append(record)

        # Convert batch to DataFrame
        batch_df = pd.DataFrame(records)
        all_dfs.append(batch_df)

        # Clear memory
        del records

    # Concatenate all batches
    print(f"  Concatenating {len(all_dfs)} batches...")
    final_df = pd.concat(all_dfs, ignore_index=True)
    del all_dfs

    return final_df


def generate_graph_seq2seq_io_data(
    df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if add_time_in_day:
        time_ind = (
            df.index.values - df.index.values.astype("datetime64[D]")
        ) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def create_hf_readme(output_dir, dataset_name):
    """Create a README.md file for Hugging Face compatibility."""

    # Determine size category based on dataset
    if dataset_name == "METR-LA":
        size_category = "1M<n<10M"  # ~7.1M records
    elif dataset_name == "PEMS-BAY":
        size_category = "10M<n<100M"  # ~16.9M records
    else:
        size_category = "1M<n<10M"  # Default fallback

    readme_content = f"""---
license: mit
task_categories:
- time-series-forecasting
- tabular-regression
tags:
- traffic-prediction
- time-series
- graph-neural-networks
- transportation
size_categories:
- {size_category}
---

# {dataset_name} Traffic Dataset

## Dataset Description

This dataset contains traffic flow data for time series forecasting tasks, commonly used with Graph Neural Networks and specifically the Diffusion Convolutional Recurrent Neural Network (DCRNN) model.

## Dataset Structure

### Data Format
- **Format**: Parquet files for efficient loading and analysis
- **Splits**: train (70%), validation (10%), test (20%) - **temporal splits** preserving chronological order
- **Features**: Time series traffic flow data with temporal and spatial dimensions

### Split Strategy
- **Temporal splitting**: Data is split chronologically to prevent data leakage
- **All sensors included**: Each split contains data for all sensors at each time step
- **Training period**: Earliest 70% of time samples across all sensors
- **Validation period**: Next 10% of time samples across all sensors  
- **Test period**: Latest 20% of time samples across all sensors
- **Graph structure preserved**: Spatial relationships maintained in all splits

### Data Schema
- `node_id`: Sensor/node identifier (0-206 for METR-LA, 0-324 for PEMS-BAY)
- `x_t*_d*`: Input features at different time offsets and dimensions
  - `x_t-11_d0` to `x_t+0_d0`: Traffic flow values at 12 historical time steps
  - `x_t-11_d1` to `x_t+0_d1`: Time-of-day features (normalized 0-1)
- `y_t*_d*`: Target values at future time steps and dimensions
  - `y_t+1_d0` to `y_t+12_d0`: Traffic flow predictions for next 12 time steps
  - `y_t+1_d1` to `y_t+12_d1`: Time-of-day features for prediction horizon

### Dataset Statistics
- **Total time series samples**: ~34K (METR-LA) / ~52K (PEMS-BAY)
- **Total records**: ~7M (METR-LA) / ~17M (PEMS-BAY) 
- **Records per sample**: 207 (METR-LA) / 325 (PEMS-BAY) sensors
- **Temporal resolution**: 5-minute intervals
- **Prediction horizon**: 1 hour (12 time steps)

## Usage

```python
from datasets import load_dataset
import pandas as pd

# Load from Hugging Face Hub
dataset = load_dataset("witgaw/{dataset_name}")

# Or load locally
train_df = pd.read_parquet("train.parquet")
val_df = pd.read_parquet("val.parquet") 
test_df = pd.read_parquet("test.parquet")

# Get number of sensors for this dataset
num_sensors = 207 if dataset_name == "METR-LA" else 325

print(f"Train samples: {{len(train_df) // num_sensors:,}}")  # Divide by number of sensors
print(f"Total records: {{len(train_df):,}}")
print(f"Features per record: {{len(train_df.columns)}}")

# Example: Get data for first time sample
first_sample = train_df[train_df.index < num_sensors]  # First N records (all sensors)
print(f"Shape for one time sample: {{first_sample.shape}}")
```

## Citation

If you use this dataset, please cite the original DCRNN paper:

```bibtex
@inproceedings{{li2018dcrnn,
  title={{Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting}},
  author={{Li, Yaguang and Yu, Rose and Shahabi, Cyrus and Liu, Yan}},
  booktitle={{International Conference on Learning Representations}},
  year={{2018}}
}}
```

## Dataset Generation

The code used to generate this Hugging Face-compatible dataset can be found at [witgaw/DCRNN](https://github.com/witgaw/DCRNN), a fork of the original DCRNN repository with enhanced data processing capabilities.

## Original Data Source

This dataset is derived from the original {dataset_name} dataset used in the DCRNN paper.

## License

MIT License - See LICENSE file for details.
"""

    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"Created README.md: {readme_path}")


def get_dataset_name_from_filename(filename):
    """Extract dataset name from the input filename."""
    basename = os.path.basename(filename)
    if "metr-la" in basename.lower():
        return "METR-LA"
    elif "pems-bay" in basename.lower():
        return "PEMS-BAY"
    else:
        # Fallback: use the filename without extension
        return os.path.splitext(basename)[0].upper()


def ensure_output_directory(base_dir, dataset_name):
    """Create output directory structure and return the path."""
    output_dir = os.path.join(base_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def generate_train_val_test(args):
    # Determine dataset name and create output directory
    dataset_name = get_dataset_name_from_filename(args.traffic_df_filename)
    output_dir = ensure_output_directory(args.output_dir, dataset_name)

    print(f"Processing dataset: {dataset_name}")
    print(f"Output directory: {output_dir}")

    # Copy sensor graph data
    print("Copying sensor graph data...")
    copy_sensor_graph_data(dataset_name, output_dir)

    df = pd.read_hdf(args.traffic_df_filename)
    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-11, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train : num_train + num_val],
        y[num_train : num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)

        if args.format == "parquet":
            # Convert arrays to DataFrame
            print(f"Converting {cat} split to DataFrame...")
            df = arrays_to_dataframe_batched(_x, _y, x_offsets, y_offsets)

            # Save as Parquet
            parquet_filename = os.path.join(output_dir, f"{cat}.parquet")
            print(f"Saving {len(df):,} records to {parquet_filename}...")
            df.to_parquet(parquet_filename, compression="snappy", index=False)

            # Also save metadata as JSON for compatibility
            metadata = {
                "x_offsets": x_offsets.tolist(),
                "y_offsets": y_offsets.tolist(),
                "x_shape": list(_x.shape),
                "y_shape": list(_y.shape),
                "num_samples": int(_x.shape[0]),
                "input_length": int(_x.shape[1]),
                "output_length": int(_y.shape[1]),
                "num_nodes": int(_x.shape[2]),
                "input_dim": int(_x.shape[3]),
                "output_dim": int(_y.shape[3]),
            }

            import json

            metadata_filename = os.path.join(output_dir, f"{cat}_metadata.json")
            with open(metadata_filename, "w") as f:
                json.dump(metadata, f, indent=2)

        # Create README.md for Hugging Face compatibility if using parquet format
        if args.format == "parquet":
            create_hf_readme(output_dir, dataset_name)

            print(f"Saved {cat}: {len(df):,} records, metadata to {metadata_filename}")

        else:  # npz format (original)
            np.savez_compressed(
                os.path.join(output_dir, "%s.npz" % cat),
                x=_x,
                y=_y,
                x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
                y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
            )


def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="data/hf_datasets/", help="Output directory."
    )
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default="data/metr-la.h5",
        help="Raw traffic readings.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["npz", "parquet"],
        default="parquet",
        help="Output format: npz (original) or parquet (HF compatible)",
    )
    args = parser.parse_args()
    main(args)
