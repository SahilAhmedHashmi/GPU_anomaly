# DATA PREPROCESSING AND COMPRESSION
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime, timedelta
from collections import defaultdict
from itertools import combinations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from tqdm import tqdm
import random
import gc
import os
import shutil
import tempfile
import threading

SEED = 42
TIME_WINDOW = 60
PREDICTION_HORIZON = 30
NUM_FEATURES = 6

START_DATE = datetime(2017, 10, 15)
END_DATE = datetime(2017, 10, 22)

@dataclass
class GPUNode:
    machine_id: str
    gpu_index: int
    node_id: str
    node_idx: int

random.seed(SEED)
np.random.seed(SEED)


_parse_errors_shown = 0

def parse_datetime(dt_str):
    global _parse_errors_shown

    if dt_str is None or pd.isna(dt_str):
        return None

    try:
        return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            cleaned = dt_str.split(" PDT")[0].split(" PST")[0]
            return datetime.strptime(cleaned, "%Y-%m-%d %H:%M:%S")
        except (ValueError, AttributeError) as e:
            if _parse_errors_shown < 5:
                print(f"Failed to parse timestamp: '{dt_str}' - {e}")
                _parse_errors_shown += 1
            return None


def validate_gpu_util_data(df, expected_cols=10):
    if len(df.columns) != expected_cols:
        raise ValueError(f"Expected {expected_cols} columns, got {len(df.columns)}")

    if df['time'].isnull().any():
        null_count = df['time'].isnull().sum()
        print(f"Warning: {null_count} rows with null timestamps will be dropped.")

    gpu_cols = [col for col in df.columns if 'util' in col.lower()]
    for col in gpu_cols:
        valid_data = pd.to_numeric(df[col], errors='coerce')
        invalid_count = ((valid_data < 0) | (valid_data > 100)).sum()
        if invalid_count > 0:
            print(f"Warning: {invalid_count} invalid values in {col} (outside 0-100 range).")

    return df


def load_job_log(filepath):
    jobs = []
    with open(filepath, 'r') as f:
        all_jobs = json.load(f)

    print(f"Loaded {len(all_jobs)} raw jobs from file.")

    for job_data in all_jobs:
        if not isinstance(job_data, dict):
            continue

        attempts = []
        for attempt in job_data.get('attempts', []):
            start_time = parse_datetime(attempt.get('start_time'))
            end_time = parse_datetime(attempt.get('end_time'))

            if start_time is None or end_time is None:
                continue

            detail = attempt.get('detail', [])
            if not detail:
                continue

            attempts.append({
                'start_time': start_time,
                'end_time': end_time,
                'detail': detail
            })

        if not attempts:
            continue

        if job_data.get('status') not in ['Pass', 'Failed', 'Killed']:
            continue

        jobs.append({
            'job_id': job_data.get('jobid'),
            'status': job_data.get('status'),
            'submitted_time': parse_datetime(job_data.get('submitted_time')),
            'attempts': attempts,
            'user': job_data.get('user'),
            'vc': job_data.get('vc')
        })

    return jobs


def load_gpu_util(filepath, chunksize=500000):
    usecols = list(range(10))
    col_names = ['time', 'machineId', 'gpu0_util', 'gpu1_util', 'gpu2_util',
                 'gpu3_util', 'gpu4_util', 'gpu5_util', 'gpu6_util', 'gpu7_util']

    chunks = []
    for chunk in tqdm(pd.read_csv(filepath, chunksize=chunksize, low_memory=False,
                                  usecols=usecols, names=col_names, header=0), desc="Loading GPU util"):
        chunk.columns = chunk.columns.str.strip()
        if 'time' in chunk.columns:
            chunk['time'] = chunk['time'].apply(
                lambda x: parse_datetime(str(x).split(" P")[0] if pd.notna(x) else None)
            )
            chunk = chunk.dropna(subset=['time'])
            chunk = chunk[(chunk['time'] >= START_DATE) & (chunk['time'] <= END_DATE)]

        if len(chunk) > 0:
            chunk['machineId'] = chunk['machineId'].astype('category')
            for col in ['gpu0_util', 'gpu1_util', 'gpu2_util', 'gpu3_util',
                        'gpu4_util', 'gpu5_util', 'gpu6_util', 'gpu7_util']:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce').astype('float32')
            chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    df = validate_gpu_util_data(df)
    print(f"Loaded GPU util data: {len(df)} rows ({START_DATE.date()} to {END_DATE.date()}).")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB.")
    return df


def filter_jobs_to_range(jobs, start_date, end_date):
    def job_in_range(job):
        for attempt in job['attempts']:
            if attempt['start_time'] and attempt['end_time']:
                if attempt['start_time'] <= end_date and attempt['end_time'] >= start_date:
                    return True
        return False

    return [j for j in jobs if job_in_range(j)]


def build_node_list_from_gpu_util(gpu_util_df):
    machine_col = None
    for col in gpu_util_df.columns:
        if 'machine' in col.lower():
            machine_col = col
            break

    if machine_col is None:
        machine_col = gpu_util_df.columns[1]

    gpu_cols = [col for col in gpu_util_df.columns if col.startswith('gpu') and 'util' in col.lower()]

    print("Counting valid GPUs per machine...")
    machine_gpu_counts = {}

    grouped = gpu_util_df.groupby(machine_col)
    for machine_id, machine_data in tqdm(grouped, desc="Processing machines"):
        valid_gpus = 0
        for gpu_col in gpu_cols:
            col_data = machine_data[gpu_col]
            non_na_count = col_data.apply(lambda x: x != 'NA' and pd.notna(x)).sum()
            if non_na_count > len(machine_data) * 0.1:
                valid_gpus += 1

        if valid_gpus > 0:
            machine_gpu_counts[machine_id] = valid_gpus

    nodes = []
    node_to_idx = {}
    idx = 0

    for machine_id, gpu_count in sorted(machine_gpu_counts.items()):
        for gpu_idx in range(gpu_count):
            node_id = f"{machine_id}_gpu{gpu_idx}"
            nodes.append(GPUNode(
                machine_id=machine_id,
                gpu_index=gpu_idx,
                node_id=node_id,
                node_idx=idx
            ))
            node_to_idx[node_id] = idx
            idx += 1

    machine_to_nodes = defaultdict(list)
    for node in nodes:
        machine_to_nodes[node.machine_id].append(node.node_idx)

    return nodes, node_to_idx, dict(machine_to_nodes), gpu_cols, machine_col


def build_intra_server_edges(machine_to_nodes):
    edges = []
    for machine_id, node_indices in machine_to_nodes.items():
        if len(node_indices) < 2:
            continue
        for i, j in combinations(node_indices, 2):
            edges.append((i, j))
            edges.append((j, i))
    return edges


def build_inter_server_edges_for_job(job, node_to_idx):
    edges = []
    for attempt in job['attempts']:
        server_to_gpus = defaultdict(list)
        for detail in attempt['detail']:
            machine_id = detail['ip']
            for gpu_name in detail['gpus']:
                gpu_idx = int(gpu_name.replace('gpu', ''))
                node_id = f"{machine_id}_gpu{gpu_idx}"
                if node_id in node_to_idx:
                    server_to_gpus[machine_id].append(node_to_idx[node_id])

        servers = list(server_to_gpus.keys())
        if len(servers) > 1:
            for i in range(len(servers)):
                for j in range(i + 1, len(servers)):
                    for gpu_a in server_to_gpus[servers[i]]:
                        for gpu_b in server_to_gpus[servers[j]]:
                            edges.append((gpu_a, gpu_b))
                            edges.append((gpu_b, gpu_a))
    return edges


def build_gpu_job_history(jobs, node_to_idx):
    gpu_history = defaultdict(list)
    for job in jobs:
        is_failed = 1 if job['status'] == 'Failed' else 0
        for attempt in job['attempts']:
            end_time = attempt.get('end_time')
            if end_time is None:
                continue
            for detail in attempt['detail']:
                machine_id = detail['ip']
                for gpu_name in detail['gpus']:
                    gpu_idx = int(gpu_name.replace('gpu', ''))
                    node_id = f"{machine_id}_gpu{gpu_idx}"
                    if node_id in node_to_idx:
                        gpu_history[node_id].append((end_time, is_failed))

    for node_id in gpu_history:
        gpu_history[node_id].sort(key=lambda x: x[0])

    return dict(gpu_history)


def build_numpy_index(gpu_util_df, nodes, node_to_idx, machine_col, gpu_cols):
    print("Building numpy index structures...")

    unique_timestamps = sorted(gpu_util_df['time'].unique())
    timestamp_to_idx = {ts: idx for idx, ts in enumerate(unique_timestamps)}
    all_timestamps = np.array(unique_timestamps)

    num_timestamps = len(unique_timestamps)
    num_nodes = len(nodes)

    print(f"  Timestamps: {num_timestamps}, Nodes: {num_nodes}")

    gpu_util_array = np.zeros((num_timestamps, num_nodes), dtype=np.float32)
    is_offline_array = np.ones((num_timestamps, num_nodes), dtype=np.float32)

    machine_gpu_to_node = {}
    for node in nodes:
        machine_gpu_to_node[(node.machine_id, node.gpu_index)] = node.node_idx

    gpu_cols_set = set(gpu_cols)

    print("  Filling arrays...")
    total_timestamps = len(gpu_util_df['time'].unique())
    for timestamp, group in tqdm(gpu_util_df.groupby('time'),
                                 desc="Building index",
                                 total=total_timestamps):
        t_idx = timestamp_to_idx[timestamp]

        for _, row in group.iterrows():
            machine_id = row[machine_col]

            for gpu_idx in range(8):
                gpu_col = f"gpu{gpu_idx}_util"

                if gpu_col not in gpu_cols_set:
                    continue

                key = (machine_id, gpu_idx)
                if key not in machine_gpu_to_node:
                    continue

                node_idx = machine_gpu_to_node[key]
                val = row[gpu_col]

                if val == 'NA' or pd.isna(val):
                    gpu_util_array[t_idx, node_idx] = 0.0
                    is_offline_array[t_idx, node_idx] = 1
                else:
                    try:
                        gpu_util_array[t_idx, node_idx] = float(val)
                        is_offline_array[t_idx, node_idx] = 0
                    except:
                        gpu_util_array[t_idx, node_idx] = 0.0
                        is_offline_array[t_idx, node_idx] = 1

    print(f"  Index built. Memory: {(gpu_util_array.nbytes + is_offline_array.nbytes) / 1e9:.2f} GB.")

    return timestamp_to_idx, all_timestamps, gpu_util_array, is_offline_array


def build_job_interval_index(jobs):
    print("Building job interval index...")

    intervals = []
    for job in jobs:
        for attempt_idx, attempt in enumerate(job['attempts']):
            if attempt['start_time'] and attempt['end_time']:
                intervals.append({
                    'start': attempt['start_time'],
                    'end': attempt['end_time'],
                    'job': job,
                    'attempt': attempt,
                    'attempt_idx': attempt_idx
                })

    intervals.sort(key=lambda x: x['start'])
    print(f"  Indexed {len(intervals)} job intervals.")
    return intervals


def get_active_jobs_fast(job_intervals, time_point):
    active_jobs = []
    for interval in job_intervals:
        if interval['start'] > time_point:
            break
        if interval['start'] <= time_point <= interval['end']:
            active_jobs.append({
                'job': interval['job'],
                'attempt': interval['attempt'],
                'attempt_idx': interval['attempt_idx']
            })
    return active_jobs


def get_gpu_historical_fail_rate(node_id, current_time, gpu_history, lookback_hours=24):
    if node_id not in gpu_history:
        return 0.0

    past_events = [(t, fail) for t, fail in gpu_history[node_id] if t < current_time]

    if len(past_events) == 0:
        return 0.0

    cutoff_time = current_time - timedelta(hours=lookback_hours)
    recent_events = [(t, fail) for t, fail in past_events if t >= cutoff_time]

    if len(recent_events) == 0:
        recent_events = past_events

    failure_count = sum(fail for _, fail in recent_events)
    return failure_count / len(recent_events)


def extract_features_vectorized(
    time_point, nodes, node_to_idx, timestamp_to_idx, all_timestamps,
    gpu_util_array, is_offline_array, gpu_history, active_jobs, time_window
):
    num_nodes = len(nodes)
    features = np.zeros((num_nodes, time_window, NUM_FEATURES), dtype=np.float32)

    start_time = time_point - timedelta(minutes=time_window)

    start_idx = np.searchsorted(all_timestamps, start_time)
    end_idx = np.searchsorted(all_timestamps, time_point, side='right')

    if start_idx >= end_idx:
        return None

    timestamps_in_window = all_timestamps[start_idx:end_idx]
    t_indices = np.arange(start_idx, end_idx)

    if len(timestamps_in_window) < time_window // 2:
        return None

    if len(timestamps_in_window) > time_window:
        timestamps_in_window = timestamps_in_window[-time_window:]
        t_indices = t_indices[-time_window:]

    gpu_to_job_info = {}
    for aj in active_jobs:
        job = aj['job']
        attempt = aj['attempt']
        attempt_idx = aj['attempt_idx']
        attempt_number = attempt_idx + 1
        job_duration_so_far = (time_point - attempt['start_time']).total_seconds() / 60.0

        for detail in attempt['detail']:
            machine_id = detail['ip']
            for gpu_name in detail['gpus']:
                gpu_idx = int(gpu_name.replace('gpu', ''))
                node_id = f"{machine_id}_gpu{gpu_idx}"
                if node_id in node_to_idx:
                    gpu_to_job_info[node_id] = {
                        'is_allocated': 1,
                        'attempt_number': attempt_number,
                        'job_duration_so_far': job_duration_so_far
                    }

    num_timestamps_to_use = len(timestamps_in_window)
    offset = time_window - num_timestamps_to_use if num_timestamps_to_use < time_window else 0

    util_slice = gpu_util_array[t_indices, :]
    offline_slice = is_offline_array[t_indices, :]

    features[:, offset:offset+num_timestamps_to_use, 0] = (util_slice / 100.0).T
    features[:, offset:offset+num_timestamps_to_use, 1] = offline_slice.T

    for node in nodes:
        node_idx = node.node_idx
        job_info = gpu_to_job_info.get(node.node_id, {
            'is_allocated': 0,
            'attempt_number': 0,
            'job_duration_so_far': 0
        })

        hist_fail_rate = get_gpu_historical_fail_rate(node.node_id, time_point, gpu_history)

        features[node_idx, offset:offset+num_timestamps_to_use, 2] = job_info['is_allocated']
        features[node_idx, offset:offset+num_timestamps_to_use, 3] = job_info['attempt_number'] / 10.0
        features[node_idx, offset:offset+num_timestamps_to_use, 4] = hist_fail_rate
        features[node_idx, offset:offset+num_timestamps_to_use, 5] = job_info['job_duration_so_far'] / 1440.0

    return features


def generate_labels_at_time(time_point, nodes, node_to_idx, jobs, prediction_horizon):
    labels = np.zeros(len(nodes), dtype=np.float32)
    horizon_end = time_point + timedelta(minutes=prediction_horizon)

    for job in jobs:
        if job['status'] != 'Failed':
            continue
        for attempt in job['attempts']:
            if attempt['start_time'] <= time_point and time_point <= attempt['end_time'] <= horizon_end:
                for detail in attempt['detail']:
                    machine_id = detail['ip']
                    for gpu_name in detail['gpus']:
                        gpu_idx = int(gpu_name.replace('gpu', ''))
                        node_id = f"{machine_id}_gpu{gpu_idx}"
                        if node_id in node_to_idx:
                            labels[node_to_idx[node_id]] = 1.0
    return labels


def create_sample_vectorized(
    time_point, nodes, node_to_idx, machine_to_nodes, timestamp_to_idx,
    all_timestamps, gpu_util_array, is_offline_array, gpu_history,
    job_intervals, jobs, precomputed_intra_edges, time_window=60, prediction_horizon=30
):
    active_jobs = get_active_jobs_fast(job_intervals, time_point)
    if len(active_jobs) == 0:
        return None

    features = extract_features_vectorized(
        time_point=time_point,
        nodes=nodes,
        node_to_idx=node_to_idx,
        timestamp_to_idx=timestamp_to_idx,
        all_timestamps=all_timestamps,
        gpu_util_array=gpu_util_array,
        is_offline_array=is_offline_array,
        gpu_history=gpu_history,
        active_jobs=active_jobs,
        time_window=time_window
    )

    if features is None:
        return None

    intra_edges = precomputed_intra_edges

    inter_edges = []
    for aj in active_jobs:
        inter_edges.extend(build_inter_server_edges_for_job(aj['job'], node_to_idx))

    all_edges = list(set(intra_edges + inter_edges))

    if len(all_edges) == 0:
        return None

    edge_index = np.array(all_edges, dtype=np.int64).T

    same_job_pairs = set()
    for aj in active_jobs:
        job_gpus = []
        for detail in aj['attempt']['detail']:
            machine_id = detail['ip']
            for gpu_name in detail['gpus']:
                gpu_idx = int(gpu_name.replace('gpu', ''))
                node_id = f"{machine_id}_gpu{gpu_idx}"
                if node_id in node_to_idx:
                    job_gpus.append(node_to_idx[node_id])
        for i, j in combinations(job_gpus, 2):
            same_job_pairs.add((i, j))
            same_job_pairs.add((j, i))

    edge_attr = []
    intra_set = set(intra_edges)
    inter_set = set(inter_edges)
    for e in all_edges:
        is_intra = 1 if e in intra_set else 0
        is_inter = 1 if e in inter_set else 0
        is_same_job = 1 if e in same_job_pairs else 0
        edge_attr.append([is_intra, is_inter, is_same_job])
    edge_attr = np.array(edge_attr, dtype=np.float32)

    labels = generate_labels_at_time(
        time_point=time_point,
        nodes=nodes,
        node_to_idx=node_to_idx,
        jobs=jobs,
        prediction_horizon=prediction_horizon
    )

    return {
        'features': features,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'labels': labels,
        'time_point': time_point,
        'num_active_jobs': len(active_jobs)
    }


def process_batch(batch_start, batch_end, batch_time_points, nodes, node_to_idx,
                  machine_to_nodes, gpu_cols, machine_col, jobs, gpu_history,
                  precomputed_intra_edges):
    print(f"\nProcessing batch: {batch_start.date()} to {batch_end.date()}")
    print(f"Time points in batch: {len(batch_time_points)}")

    if len(batch_time_points) == 0:
        return []

    data_start = batch_start - timedelta(minutes=TIME_WINDOW)
    data_end = batch_end

    print(f"Loading GPU util from {data_start} to {data_end} ({TIME_WINDOW}min overlap)...")

    usecols = list(range(10))
    col_names = ['time', 'machineId', 'gpu0_util', 'gpu1_util', 'gpu2_util',
                 'gpu3_util', 'gpu4_util', 'gpu5_util', 'gpu6_util', 'gpu7_util']

    chunks = []
    for chunk in tqdm(pd.read_csv(GPU_UTIL_PATH, chunksize=500000, low_memory=False,
                                  usecols=usecols, names=col_names, header=0), desc="Loading"):
        chunk.columns = chunk.columns.str.strip()
        if 'time' in chunk.columns:
            chunk['time'] = chunk['time'].apply(
                lambda x: parse_datetime(str(x).split(" P")[0] if pd.notna(x) else None)
            )
            chunk = chunk.dropna(subset=['time'])
            chunk = chunk[(chunk['time'] >= data_start) & (chunk['time'] <= data_end)]

        if len(chunk) > 0:
            chunk['machineId'] = chunk['machineId'].astype('category')
            for col in ['gpu0_util', 'gpu1_util', 'gpu2_util', 'gpu3_util',
                        'gpu4_util', 'gpu5_util', 'gpu6_util', 'gpu7_util']:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce').astype('float32')
            chunks.append(chunk)

    if len(chunks) == 0:
        print("No data for this batch.")
        return []

    gpu_util_df = pd.concat(chunks, ignore_index=True)
    print(f"Batch GPU util rows: {len(gpu_util_df)}")
    del chunks
    gc.collect()

    print("Building batch index...")
    timestamp_to_idx, all_timestamps, gpu_util_array, is_offline_array = build_numpy_index(
        gpu_util_df, nodes, node_to_idx, machine_col, gpu_cols
    )
    del gpu_util_df
    gc.collect()

    job_intervals = build_job_interval_index(jobs)

    samples = []
    for tp in tqdm(batch_time_points, desc="Creating samples"):
        sample = create_sample_vectorized(
            time_point=tp,
            nodes=nodes,
            node_to_idx=node_to_idx,
            machine_to_nodes=machine_to_nodes,
            timestamp_to_idx=timestamp_to_idx,
            all_timestamps=all_timestamps,
            gpu_util_array=gpu_util_array,
            is_offline_array=is_offline_array,
            gpu_history=gpu_history,
            job_intervals=job_intervals,
            jobs=jobs,
            precomputed_intra_edges=precomputed_intra_edges,
            time_window=TIME_WINDOW,
            prediction_horizon=PREDICTION_HORIZON
        )
        if sample is not None:
            samples.append(sample)

    print(f"Batch samples created: {len(samples)}")

    del gpu_util_array, is_offline_array, all_timestamps
    gc.collect()

    return samples


def get_checkpoint_metadata():
    return {
        'time_window': TIME_WINDOW,
        'prediction_horizon': PREDICTION_HORIZON,
        'num_features': NUM_FEATURES,
        'start_date': str(START_DATE),
        'end_date': str(END_DATE),
        'seed': SEED
    }


def save_batch_checkpoint(batch_samples, checkpoint_file, num_nodes):
    checkpoint_data = {
        'samples': batch_samples,
        'metadata': get_checkpoint_metadata(),
        'num_nodes': num_nodes,
        'version': 1
    }
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"  Saved checkpoint: {len(batch_samples)} samples.")


def load_batch_checkpoint(checkpoint_file, expected_num_nodes):
    try:
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
    except Exception as e:
        print(f"  Failed to load checkpoint: {e}")
        return None

    saved_metadata = checkpoint_data.get('metadata', {})
    current_metadata = get_checkpoint_metadata()

    mismatches = []
    for key in current_metadata:
        if saved_metadata.get(key) != current_metadata[key]:
            mismatches.append(f"{key}: {saved_metadata.get(key)} -> {current_metadata[key]}")

    if checkpoint_data.get('num_nodes') != expected_num_nodes:
        mismatches.append(f"num_nodes: {checkpoint_data.get('num_nodes')} -> {expected_num_nodes}")

    if mismatches:
        print("  Checkpoint config mismatch:")
        for m in mismatches:
            print(f"    {m}")
        return None

    print("  Checkpoint validated.")
    return checkpoint_data['samples']


def validate_samples(samples, expected_nodes=104, expected_timesteps=60,
                     expected_features=6, sample_size=100):
    print("\nValidating samples...")

    indices_to_check = list(range(min(sample_size, len(samples))))
    if len(samples) > sample_size:
        middle = len(samples) // 2
        indices_to_check.extend(range(middle - 10, middle + 10))
        indices_to_check.extend(range(len(samples) - 20, len(samples)))

    indices_to_check = sorted(set([i for i in indices_to_check if 0 <= i < len(samples)]))

    issues = []
    for i in indices_to_check:
        sample = samples[i]

        if sample['features'].shape != (expected_nodes, expected_timesteps, expected_features):
            issues.append(f"Sample {i}: Invalid feature shape {sample['features'].shape}")

        if sample['edge_index'].shape[0] != 2:
            issues.append(f"Sample {i}: Invalid edge_index shape {sample['edge_index'].shape}")

        if sample['labels'].shape != (expected_nodes,):
            issues.append(f"Sample {i}: Invalid labels shape {sample['labels'].shape}")

        if not (0 <= sample['features'][:, :, 0].max() <= 1.01):
            issues.append(f"Sample {i}: GPU utilization not normalized (max={sample['features'][:, :, 0].max():.2f})")

        if not np.all(np.isin(sample['labels'], [0, 1])):
            issues.append(f"Sample {i}: Labels not binary")

        if np.isnan(sample['features']).any():
            issues.append(f"Sample {i}: Contains NaN in features")
        if np.isinf(sample['features']).any():
            issues.append(f"Sample {i}: Contains Inf in features")

    if issues:
        print(f"Found {len(issues)} validation issues:")
        for issue in issues[:10]:
            print(f"  {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
        raise ValueError("Sample validation failed.")
    else:
        print(f"Validated {len(indices_to_check)} samples successfully.")


def report_memory_usage(samples):
    if len(samples) == 0:
        print("No samples to report.")
        return

    sample = samples[0]

    feature_bytes = sample['features'].nbytes
    edge_index_bytes = sample['edge_index'].nbytes
    edge_attr_bytes = sample['edge_attr'].nbytes
    labels_bytes = sample['labels'].nbytes

    per_sample = feature_bytes + edge_index_bytes + edge_attr_bytes + labels_bytes
    total_bytes = per_sample * len(samples)

    print(f"\nMemory Usage:")
    print(f"  Features:       {feature_bytes / 1e6:.2f} MB per sample")
    print(f"  Edge Index:     {edge_index_bytes / 1e6:.2f} MB per sample")
    print(f"  Edge Attr:      {edge_attr_bytes / 1e6:.2f} MB per sample")
    print(f"  Labels:         {labels_bytes / 1e6:.2f} MB per sample")
    print(f"  Per sample:     {per_sample / 1e6:.2f} MB")
    print(f"  Total dataset:  {total_bytes / 1e9:.2f} GB")


def report_class_balance(samples, split_name):
    if len(samples) == 0:
        print(f"\n{split_name}: No samples.")
        return

    expected_nodes = samples[0]['labels'].shape[0]
    assert all(s['labels'].shape[0] == expected_nodes for s in samples), \
        f"Not all samples have {expected_nodes} nodes in {split_name} split."

    total_nodes = len(samples) * expected_nodes
    total_failures = sum(s['labels'].sum() for s in samples)
    failure_rate = total_failures / total_nodes

    samples_with_failures = sum(1 for s in samples if s['labels'].sum() > 0)
    failure_sample_rate = samples_with_failures / len(samples)

    print(f"\n{split_name} Class Balance:")
    print(f"  Samples:              {len(samples)}")
    print(f"  Samples with failures:{samples_with_failures} ({failure_sample_rate:.1%})")
    print(f"  Total nodes:          {total_nodes:,}")
    print(f"  Failing nodes:        {int(total_failures)}")
    print(f"  Failure rate:         {failure_rate:.1%}")


print("Loading all jobs...")
all_jobs = load_job_log(JOB_LOG_PATH)
print(f"Total jobs: {len(all_jobs)}")

print("Filtering jobs to date range...")
jobs = filter_jobs_to_range(all_jobs, START_DATE, END_DATE)
print(f"Jobs in range: {len(jobs)}")
print(f"  Failed: {len([j for j in jobs if j['status'] == 'Failed'])}")
print(f"  Passed: {len([j for j in jobs if j['status'] == 'Pass'])}")

print("\nBuilding node list from GPU util sample...")
sample_chunks = []
rows_loaded = 0
target_rows = 2000000

for chunk in tqdm(pd.read_csv(GPU_UTIL_PATH, chunksize=500000, low_memory=False,
                              usecols=list(range(10)),
                              names=['time', 'machineId', 'gpu0_util', 'gpu1_util', 'gpu2_util',
                                     'gpu3_util', 'gpu4_util', 'gpu5_util', 'gpu6_util', 'gpu7_util'],
                              header=0), desc="Loading sample"):
    chunk.columns = chunk.columns.str.strip()
    sample_chunks.append(chunk)
    rows_loaded += len(chunk)
    if rows_loaded >= target_rows:
        break

sample_df = pd.concat(sample_chunks, ignore_index=True)
del sample_chunks
print(f"Loaded {len(sample_df)} rows for node detection.")

nodes, node_to_idx, machine_to_nodes, gpu_cols, machine_col = build_node_list_from_gpu_util(sample_df)
del sample_df
gc.collect()

print(f"GPU nodes: {len(nodes)}")
print(f"Machines: {len(machine_to_nodes)}")

print("\nBuilding GPU job history from all jobs...")
gpu_history = build_gpu_job_history(all_jobs, node_to_idx)
print(f"GPUs with history: {len(gpu_history)}")

del all_jobs
gc.collect()

print("\nPre-computing intra-server edges...")
precomputed_intra_edges = build_intra_server_edges(machine_to_nodes)
print(f"Intra edges: {len(precomputed_intra_edges)}")

print("\nScanning timestamp range...")
min_time = None
max_time = None
for chunk in tqdm(pd.read_csv(GPU_UTIL_PATH, chunksize=1000000, low_memory=False,
                              usecols=[0], names=['time'], header=0), desc="Scanning"):
    chunk['time'] = chunk['time'].apply(
        lambda x: parse_datetime(str(x).split(" P")[0] if pd.notna(x) else None)
    )
    chunk = chunk.dropna()
    chunk_times = chunk['time']
    chunk_times = chunk_times[(chunk_times >= START_DATE) & (chunk_times <= END_DATE)]

    if len(chunk_times) > 0:
        if min_time is None:
            min_time = chunk_times.min()
            max_time = chunk_times.max()
        else:
            min_time = min(min_time, chunk_times.min())
            max_time = max(max_time, chunk_times.max())

print(f"Time range: {min_time} to {max_time}")

sample_start = min_time + timedelta(minutes=TIME_WINDOW)
sample_end = max_time - timedelta(minutes=PREDICTION_HORIZON)

failed_jobs = [j for j in jobs if j['status'] == 'Failed']
print(f"Failed jobs: {len(failed_jobs)}")

time_points = set()
for job in failed_jobs:
    for attempt in job['attempts']:
        end_time = attempt['end_time']
        if sample_start <= end_time <= sample_end:
            for offset in [-30, -15, -10, -5, 0]:
                tp = end_time + timedelta(minutes=offset)
                if sample_start <= tp <= sample_end:
                    time_points.add(tp)

print(f"Time points from failed jobs: {len(time_points)}")

passed_jobs = [j for j in jobs if j['status'] == 'Pass']
num_negative_samples = min(len(time_points) * 3, len(passed_jobs))

for job in random.sample(passed_jobs, num_negative_samples):
    for attempt in job['attempts']:
        mid_time = attempt['start_time'] + (attempt['end_time'] - attempt['start_time']) / 2
        if sample_start <= mid_time <= sample_end:
            time_points.add(mid_time)
            break

print(f"Total time points: {len(time_points)}")
all_time_points = sorted(list(time_points))

print("\nProcessing batches...")

batches = []
current = START_DATE
while current < END_DATE:
    batch_end = min(current + timedelta(days=1), END_DATE)
    batches.append((current, batch_end))
    current = batch_end

print(f"Total batches: {len(batches)}")

batch_time_points = {i: [] for i in range(len(batches))}
for tp in all_time_points:
    for i, (batch_start, batch_end) in enumerate(batches):
        if batch_start <= tp < batch_end:
            batch_time_points[i].append(tp)
            break

all_samples = []
checkpoint_dir = "batch_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

for i, (batch_start, batch_end) in enumerate(batches):
    print(f"\nBatch {i+1}/{len(batches)}")

    checkpoint_file = os.path.join(checkpoint_dir, f"batch_{i}.pkl")

    if os.path.exists(checkpoint_file):
        print(f"  Checking checkpoint for batch {i+1}...")
        batch_samples = load_batch_checkpoint(checkpoint_file, len(nodes))

        if batch_samples is not None:
            print(f"  Loaded {len(batch_samples)} samples from checkpoint.")
        else:
            print(f"  Checkpoint invalid, recomputing...")
            batch_samples = process_batch(
                batch_start, batch_end, batch_time_points[i],
                nodes, node_to_idx, machine_to_nodes, gpu_cols, machine_col,
                jobs, gpu_history, precomputed_intra_edges
            )
            save_batch_checkpoint(batch_samples, checkpoint_file, len(nodes))
    else:
        batch_samples = process_batch(
            batch_start, batch_end, batch_time_points[i],
            nodes, node_to_idx, machine_to_nodes, gpu_cols, machine_col,
            jobs, gpu_history, precomputed_intra_edges
        )
        save_batch_checkpoint(batch_samples, checkpoint_file, len(nodes))

    all_samples.extend(batch_samples)
    print(f"Total samples so far: {len(all_samples)}")
    gc.collect()

print(f"\nTotal samples: {len(all_samples)}")

validate_samples(all_samples, expected_nodes=len(nodes))
report_memory_usage(all_samples)

positive_samples = sum(1 for s in all_samples if s['labels'].sum() > 0)
print(f"Samples with positive labels: {positive_samples}")

print("\nCleaning up checkpoints...")
shutil.rmtree(checkpoint_dir)
print("Checkpoints removed.")

print("\nSplitting dataset...")

all_samples.sort(key=lambda x: x['time_point'])

n_train = int(len(all_samples) * 0.7)
n_val = int(len(all_samples) * 0.15)

train_samples = all_samples[:n_train]
val_samples = all_samples[n_train:n_train + n_val]
test_samples = all_samples[n_train + n_val:]

print(f"Train: {train_samples[0]['time_point']} to {train_samples[-1]['time_point']}")
print(f"Val:   {val_samples[0]['time_point']} to {val_samples[-1]['time_point']}")
print(f"Test:  {test_samples[0]['time_point']} to {test_samples[-1]['time_point']}")

assert train_samples[-1]['time_point'] < val_samples[0]['time_point'], "Train/Val overlap."
assert val_samples[-1]['time_point'] < test_samples[0]['time_point'], "Val/Test overlap."
print("Temporal split validated.")

report_class_balance(train_samples, "Train")
report_class_balance(val_samples, "Val")
report_class_balance(test_samples, "Test")

temp_path = OUTPUT_PATH + '.tmp'

print(f"\nSaving to {OUTPUT_PATH}...")
try:
    with open(temp_path, 'wb') as f:
        pickle.dump({
            'train_samples': train_samples,
            'val_samples': val_samples,
            'test_samples': test_samples,
            'nodes': nodes,
            'node_to_idx': node_to_idx,
            'config': {
                'time_window': TIME_WINDOW,
                'prediction_horizon': PREDICTION_HORIZON,
                'num_features': NUM_FEATURES,
                'start_date': str(START_DATE),
                'end_date': str(END_DATE),
                'num_nodes': len(nodes),
                'num_machines': len(machine_to_nodes)
            }
        }, f)

    shutil.move(temp_path, OUTPUT_PATH)
    print(f"Saved successfully to {OUTPUT_PATH}.")

except Exception as e:
    print(f"Error saving: {e}")
    if os.path.exists(temp_path):
        os.remove(temp_path)
    raise

print("\nPreprocessing complete.")
print(f"  Output file:   {OUTPUT_PATH}")
print(f"  File size:     {os.path.getsize(OUTPUT_PATH) / (1024**3):.2f} GB")
print(f"  Date range:    {START_DATE.date()} to {END_DATE.date()}")
print(f"  Total nodes:   {len(nodes)} GPUs across {len(machine_to_nodes)} machines")
print(f"  Train samples: {len(train_samples)} ({len(train_samples)/len(all_samples):.1%})")
print(f"  Val samples:   {len(val_samples)} ({len(val_samples)/len(all_samples):.1%})")
print(f"  Test samples:  {len(test_samples)} ({len(test_samples)/len(all_samples):.1%})")
print(f"  Total samples: {len(all_samples)}")

total_train_failures = sum(s['labels'].sum() for s in train_samples)
total_val_failures = sum(s['labels'].sum() for s in val_samples)
total_test_failures = sum(s['labels'].sum() for s in test_samples)
print(f"  Train failure rate: {total_train_failures / (len(train_samples) * len(nodes)):.1%}")
print(f"  Val failure rate:   {total_val_failures / (len(val_samples) * len(nodes)):.1%}")
print(f"  Test failure rate:  {total_test_failures / (len(test_samples) * len(nodes)):.1%}")
# END OF PREPROOCESSING PIPELINE

# MODEL INITIALIZATION AND TRAINING
import pandas as pd
import numpy as np
import modal
import os
import json
import warnings
from datetime import datetime, timedelta
from collections import defaultdict
from itertools import combinations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pickle
import gzip
from tqdm import tqdm


app = modal.App("philly-stgnn-v2")

volume = modal.Volume.from_name("philly-data", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch",
    "torch-geometric", 
    "pandas",
    "numpy",
    "scikit-learn",
    "tqdm"
)

DATA_DIR = "/data"
TIME_WINDOW = 60              
PREDICTION_HORIZON = 30       
NUM_FEATURES = 6              
HIDDEN_DIM = 64
BATCH_SIZE = 32
EPOCHS = 50
SEED = 42

SEED = 42

PREPROCESSED_FILENAME_COMPRESSED = "preprocessed_data_new.pkl.gz"
PREPROCESSED_FILENAME_UNCOMPRESSED = "preprocessed_data_new.pkl"

@dataclass
class PhillyConfig:
    time_window: int = 60
    prediction_horizon: int = 30
    num_features: int = 6
    min_job_duration_minutes: int = 5
    max_samples: int = None
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass  
class GPUNode:
    machine_id: str
    gpu_index: int
    node_id: str  
    node_idx: int  


GPU_CONFIG = "L4:1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'GPUNode':
            return GPUNode
        if name == 'PhillyConfig':
            return PhillyConfig
        return super().find_class(module, name)


def load_preprocessed_data(data_dir):
    compressed_path = os.path.join(data_dir, PREPROCESSED_FILENAME_COMPRESSED)
    uncompressed_path = os.path.join(data_dir, PREPROCESSED_FILENAME_UNCOMPRESSED)
    
    if os.path.exists(compressed_path):
        print(f"Loading compressed data from {compressed_path}...")
        with gzip.open(compressed_path, 'rb') as f:
            data = CustomUnpickler(f).load()  
            
        print("Converting float16 to float32 if needed...")
        for split in ['train_samples', 'val_samples', 'test_samples']:
            for sample in data[split]:
                if sample['features'].dtype == np.float16:
                    sample['features'] = sample['features'].astype(np.float32)
                if sample['edge_attr'].dtype == np.float16:
                    sample['edge_attr'] = sample['edge_attr'].astype(np.float32)
        
        print("Data loaded and converted successfully")
        
    elif os.path.exists(uncompressed_path):
        print(f"Loading uncompressed data from {uncompressed_path}...")
        with open(uncompressed_path, 'rb') as f:
            data = CustomUnpickler(f).load()  
        print("Data loaded successfully")
        
    else:
        raise FileNotFoundError(f"No preprocessed data found in {data_dir}. "
                               f"Looked for: {compressed_path} or {uncompressed_path}")
    
    return data


class PhillyDataset(Dataset):
    def __init__(self, samples):
        print(f"  Converting {len(samples)} samples to tensors...")
        self.samples = []
        for sample in tqdm(samples, desc="  Tensorizing"):
            self.samples.append({
                'features': torch.tensor(sample['features'], dtype=torch.float32),
                'edge_index': torch.tensor(sample['edge_index'], dtype=torch.long),
                'edge_attr': torch.tensor(sample['edge_attr'], dtype=torch.float32),
                'labels': (torch.tensor(sample['labels'], dtype=torch.float32) > 0).float()
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    data_list = []
    for item in batch:
        data = Data(
            x=item['features'],
            edge_index=item['edge_index'],
            edge_attr=item['edge_attr'],
            y=item['labels']
        )
        data_list.append(data)
    return Batch.from_data_list(data_list)


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.output_dim = hidden_dim * 2
    
    def forward(self, x):
        batch_size, num_nodes, time_steps, features = x.shape
        x = x.view(batch_size * num_nodes, time_steps, features)
        output, (h_n, c_n) = self.lstm(x)
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_combined = torch.cat([h_forward, h_backward], dim=-1)
        h_combined = h_combined.view(batch_size, num_nodes, -1)
        return h_combined


class SpatialEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=2, num_layers=2, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        self.layers.append(GATv2Conv(input_dim, hidden_dim, heads=num_heads, dropout=dropout, edge_dim=3))
        self.norms.append(nn.LayerNorm(hidden_dim * num_heads))
        
        for _ in range(num_layers - 1):
            self.layers.append(GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout, edge_dim=3))
            self.norms.append(nn.LayerNorm(hidden_dim * num_heads))
        
        self.output_dim = hidden_dim * num_heads
    
    def forward(self, x, edge_index, edge_attr):
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.elu(x)
        return x


class SpatioTemporalGNN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_heads=2, num_temporal_layers=1, num_spatial_layers=2, dropout=0.2):
        super().__init__()
        
        self.temporal_encoder = TemporalEncoder(
            input_dim=num_features,
            hidden_dim=hidden_dim,
            num_layers=num_temporal_layers,
            dropout=dropout
        )
        
        self.spatial_encoder = SpatialEncoder(
            input_dim=self.temporal_encoder.output_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_spatial_layers,
            dropout=dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.spatial_encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        batch_idx = batch.batch
        
        num_graphs = batch.num_graphs
        total_nodes = x.size(0)
        nodes_per_graph = total_nodes // num_graphs
        time_steps = x.size(1)
        features = x.size(2)
        
        x = x.view(num_graphs, nodes_per_graph, time_steps, features)
        
        temporal_out = self.temporal_encoder(x)
        
        temporal_out = temporal_out.view(-1, temporal_out.shape[-1])
        
        spatial_out = self.spatial_encoder(temporal_out, edge_index, edge_attr)
        
        logits = self.classifier(spatial_out).squeeze(-1)
        
        return logits


from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit


def compute_metrics(preds, labels, threshold=0.5):
    preds_binary = (preds >= threshold).astype(int)
    labels_binary = labels.astype(int)
    
    precision = precision_score(labels_binary, preds_binary, zero_division=0)
    recall = recall_score(labels_binary, preds_binary, zero_division=0)
    f1 = f1_score(labels_binary, preds_binary, zero_division=0)
    accuracy = (preds_binary == labels_binary).mean()
    
    try:
        auc = roc_auc_score(labels_binary, preds)
    except:
        auc = 0.0
    
    cm = confusion_matrix(labels_binary, preds_binary)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'threshold': threshold
    }


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(batch)
                loss = criterion(logits, batch.y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        preds = torch.sigmoid(logits).detach().cpu().numpy()
        labels = batch.y.detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
    
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
    
    return avg_loss, metrics

def find_best_threshold(preds, labels):
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.05, 0.95, 0.05):
        preds_binary = (preds >= threshold).astype(int)
        f1 = f1_score(labels.astype(int), preds_binary, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, find_threshold=False, threshold=None):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        batch = batch.to(device)
        logits = model(batch)
        loss = criterion(logits, batch.y)
        
        total_loss += loss.item()
        
        preds = torch.sigmoid(logits).cpu().numpy()
        labels = batch.y.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
        
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
        
    if threshold is not None:
        pass
    elif find_threshold:
        threshold = find_best_threshold(all_preds, all_labels)
    else:
        threshold = 0.5
    
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_preds, all_labels, threshold=threshold)
    
    return avg_loss, metrics
    


def setup_multi_gpu(model):
    return model

@app.function(
    image=image,
    volumes={DATA_DIR: volume},
    timeout=3600
)
def decompress_data():
    compressed_path = os.path.join(DATA_DIR, "preprocessed_data.pkl.gz")
    
    if not os.path.exists(compressed_path):
        return {'status': 'error', 'message': f'Compressed file not found: {compressed_path}'}
    
    compressed_size = os.path.getsize(compressed_path) / (1024**3)
    print(f"Compressed file size: {compressed_size:.2f} GB")
    print("Note: Data will be decompressed on-the-fly during training.")
    
    print("Verifying compressed file integrity...")
    try:
        with gzip.open(compressed_path, 'rb') as f:
            data = pickle.load(f)
            
        print(f"Verified! Contains:")
        print(f"  Train samples: {len(data['train_samples'])}")
        print(f"  Val samples: {len(data['val_samples'])}")
        print(f"  Test samples: {len(data['test_samples'])}")
        print(f"  Nodes: {len(data['nodes'])}")
        
        return {
            'status': 'success',
            'train_samples': len(data['train_samples']),
            'val_samples': len(data['val_samples']),
            'test_samples': len(data['test_samples']),
            'num_nodes': len(data['nodes'])
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


@app.function(
    image=image,
    volumes={DATA_DIR: volume},
    timeout=600
)
def check_data_files():
    files = os.listdir(DATA_DIR)
    print(f"Files in {DATA_DIR}: {files}")
    
    file_info = {}
    for f in files:
        path = os.path.join(DATA_DIR, f)
        if os.path.isfile(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            file_info[f] = f"{size_mb:.2f} MB"
    
    print(f"File sizes: {file_info}")

    has_compressed = 'preprocessed_data.pkl.gz' in files
    has_uncompressed = 'preprocessed_data.pkl' in files
    
    status = 'ready' if (has_compressed or has_uncompressed) else 'missing_preprocessed_data'
    
    return {
        'status': status,
        'files': file_info,
        'has_compressed': has_compressed,
        'has_uncompressed': has_uncompressed
    }


@app.function(
    image=image,
    volumes={DATA_DIR: volume},
    gpu=GPU_CONFIG,
    timeout=600
)
def check_gpu_setup():
    import torch
    
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count(),
        'devices': []
    }
    
    for i in range(torch.cuda.device_count()):
        info['devices'].append({
            'index': i,
            'name': torch.cuda.get_device_name(i),
            'memory_total': f"{torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB"
        })
    
    print(f"GPU Info: {info}")
    return info


@app.function(
    image=image,
    volumes={DATA_DIR: volume},
    gpu=GPU_CONFIG,
    timeout=14400,
    memory=65536  
)
def train_model(
    hidden_dim: int = 64,
    num_heads: int = 2,
    num_temporal_layers: int = 1,
    num_spatial_layers: int = 2,
    dropout: float = 0.5,
    learning_rate: float = 0.00005,
    weight_decay: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 32
    ):
    import torch
    import numpy as np
    from tqdm import tqdm
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print("\n" + "="*60)
    print("Loading preprocessed data...")
    print("="*60)
    data = load_preprocessed_data(DATA_DIR)
    
    train_samples = data['train_samples']
    val_samples = data['val_samples']
    test_samples = data['test_samples']
    num_nodes = len(data['nodes'])

    all_samples = train_samples + val_samples + test_samples
    stratify_keys = np.array([1 if s['labels'].sum() > 0 else 0 for s in all_samples])

    sss_train_rest = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, rest_idx = next(sss_train_rest.split(all_samples, stratify_keys))

    rest_keys = stratify_keys[rest_idx]
    sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_idx_local, test_idx_local = next(sss_val_test.split(rest_idx, rest_keys))

    val_idx = rest_idx[val_idx_local]
    test_idx = rest_idx[test_idx_local]

    train_samples = [all_samples[i] for i in train_idx]
    val_samples = [all_samples[i] for i in val_idx]
    test_samples = [all_samples[i] for i in test_idx]
    
    print(f"\nDataset loaded:")
    print(f"  Train: {len(train_samples)}")
    print(f"  Val: {len(val_samples)}")
    print(f"  Test: {len(test_samples)}")
    print(f"  Nodes: {num_nodes}")
    
    sample = train_samples[0]
    print(f"\nSample structure:")
    print(f"  features: {sample['features'].shape}")
    print(f"  edge_index: {sample['edge_index'].shape}")
    print(f"  edge_attr: {sample['edge_attr'].shape}")
    print(f"  labels: {sample['labels'].shape}")
    
    def analyze_samples(samples, name):
        total_gpus = 0
        total_failures = 0
        samples_with_failures = 0
        samples_all_zeros = 0
        failure_counts = []
        
        for sample in samples:
            labels = sample['labels']
            num_failures = labels.sum()
            total_gpus += len(labels)
            total_failures += num_failures
            
            if num_failures > 0:
                samples_with_failures += 1
                failure_counts.append(num_failures)
            else:
                samples_all_zeros += 1
        
        print(f"\n{name}:")
        print(f"  Total samples: {len(samples)}")
        print(f"  Samples with 0 failures: {samples_all_zeros} ({samples_all_zeros/len(samples)*100:.1f}%)")
        print(f"  Samples with ≥1 failure: {samples_with_failures} ({samples_with_failures/len(samples)*100:.1f}%)")
        print(f"  Total GPU predictions: {total_gpus}")
        print(f"  Total failures: {total_failures} ({total_failures/total_gpus*100:.2f}%)")
        if failure_counts:
            print(f"  Avg failures per affected sample: {np.mean(failure_counts):.2f}")
            print(f"  Max failures in one sample: {max(failure_counts)}")
    
    analyze_samples(train_samples, "TRAIN")
    analyze_samples(val_samples, "VAL")
    analyze_samples(test_samples, "TEST")
    
    print("\n" + "="*60)
    print("Creating data loaders...")
    print("="*60)
    
    train_dataset = PhillyDataset(train_samples)
    val_dataset = PhillyDataset(val_samples)
    test_dataset = PhillyDataset(test_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    print("\n" + "="*60)
    print("Initializing model...")
    print("="*60)
    
    model = SpatioTemporalGNN(
        num_features=NUM_FEATURES,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_temporal_layers=num_temporal_layers,
        num_spatial_layers=num_spatial_layers,
        dropout=dropout
    )
    
    model = model.to(device)
    model = setup_multi_gpu(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    pos_weight = torch.tensor([3.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Using BCEWithLogitsLoss with pos_weight={pos_weight.item():.1f}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3 )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_f1 = 0.0
    best_val_threshold = 0.5
    best_model_state = None
    patience_counter = 0
    early_stop_patience = 5
    
    history = {
        'train_loss': [], 'train_f1': [], 'train_auc': [], 'train_acc': [],
        'val_loss': [], 'val_f1': [], 'val_auc': [], 'val_acc': []
    }
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    for epoch in range(epochs):
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, find_threshold=True)
        
        scheduler.step(val_metrics['f1'])
        
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_metrics['f1'])
        history['train_auc'].append(train_metrics['auc'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Val   - Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_val_threshold = val_metrics['threshold']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  *** New best model! F1: {best_val_f1:.4f} ***")
        else:
            patience_counter += 1
        
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print("\n" + "="*60)
    print("Evaluating best model on test set...")
    print("="*60)
    
    model.load_state_dict(best_model_state)
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device, threshold=best_val_threshold)
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  Confusion Matrix:\n{test_metrics['confusion_matrix']}")
    
    model_path = os.path.join(DATA_DIR, "best_model.pt")
    torch.save({
        'model_state_dict': best_model_state,
        'config': {
            'hidden_dim': hidden_dim,
            'num_heads': num_heads,
            'num_temporal_layers': num_temporal_layers,
            'num_spatial_layers': num_spatial_layers,
            'dropout': dropout
        },
        'test_metrics': test_metrics,
        'history': history
    }, model_path)
    
    volume.commit()
    
    print(f"\nModel saved to {model_path}")
    
    return {
        'best_val_f1': best_val_f1,
        'test_f1': test_metrics['f1'],
        'test_auc': test_metrics['auc'],
        'test_accuracy': test_metrics['accuracy'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall']
    }


@app.function(
    image=image,
    volumes={DATA_DIR: volume},
    gpu=GPU_CONFIG,
    timeout=3600
)
def run_inference(sample_idx: int = 0):
    import torch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    data = load_preprocessed_data(DATA_DIR)
    
    model_path = os.path.join(DATA_DIR, "best_model.pt")
    checkpoint = torch.load(model_path, map_location=device,  weights_only=False)
    
    config = checkpoint['config']
    model = SpatioTemporalGNN(
        num_features=NUM_FEATURES,
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        num_temporal_layers=config['num_temporal_layers'],
        num_spatial_layers=config['num_spatial_layers'],
        dropout=config['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    test_samples = data['test_samples']
    sample = test_samples[sample_idx]
    
    dataset = PhillyDataset([sample])
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
    
    labels = (sample['labels'] > 0).astype(np.float32)
    
    results = {
        'sample_idx': sample_idx,
        'time_point': str(sample['time_point']),
        'num_active_jobs': sample['num_active_jobs'],
        'total_gpus': len(labels),
        'actual_failures': int(labels.sum()),
        'predicted_failures': int((probs >= 0.5).sum()),
        'max_probability': float(probs.max()),
        'mean_probability': float(probs.mean())
    }
    
    print(f"Inference Results: {results}")
    return results

@app.function(
    image=image,
    volumes={DATA_DIR: volume},
    gpu=GPU_CONFIG,
    timeout=3600
)
def batch_inference(num_samples: int = 100):
    import torch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_preprocessed_data(DATA_DIR)
    
    model_path = os.path.join(DATA_DIR, "best_model.pt")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    model = SpatioTemporalGNN(
        num_features=NUM_FEATURES,
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        num_temporal_layers=config['num_temporal_layers'],
        num_spatial_layers=config['num_spatial_layers'],
        dropout=config['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    test_samples = data['test_samples']
    num_samples = min(num_samples, len(test_samples))
    
    results = []
    samples_with_failures = []
    samples_without_failures = []
    
    print(f"Running inference on {num_samples} samples...")
    
    for idx in range(num_samples):
        sample = test_samples[idx]
        dataset = PhillyDataset([sample])
        loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = model(batch)
                probs = torch.sigmoid(logits).cpu().numpy()
        
        labels = (sample['labels'] > 0).astype(np.float32)
        actual_failures = int(labels.sum())
        predicted_failures = int((probs >= 0.5).sum())
        
        result = {
            'sample_idx': idx,
            'time_point': str(sample['time_point']),
            'actual_failures': actual_failures,
            'predicted_failures': predicted_failures,
            'correct': (actual_failures > 0) == (predicted_failures > 0)
        }
        
        if actual_failures > 0:
            samples_with_failures.append(result)
        else:
            samples_without_failures.append(result)
    
    total_correct = sum(1 for r in (samples_with_failures + samples_without_failures) if r['correct'])
    
    summary = {
        'total_samples': num_samples,
        'samples_with_failures': len(samples_with_failures),
        'samples_without_failures': len(samples_without_failures),
        'overall_accuracy': total_correct / num_samples,
        'examples_with_failures': samples_with_failures[:5],
        'examples_without_failures': samples_without_failures[:5]
    }
    
    print(f"\nSummary: {num_samples} samples, {len(samples_with_failures)} with failures")
    return summary

@app.function(
    image=image,
    volumes={DATA_DIR: volume},
    timeout=600
)
def get_training_history():
    import torch
    
    model_path = os.path.join(DATA_DIR, "best_model.pt")
    
    if not os.path.exists(model_path):
        return {'error': 'Model not found. Run training first.'}
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    return {
        'history': checkpoint['history'],
        'test_metrics': checkpoint['test_metrics'],
        'config': checkpoint['config']
    }

@app.local_entrypoint()
def main(
    action: str = "check",
    epochs: int = 50,
    batch_size: int = 32,
    hidden_dim: int = 64,
    learning_rate: float = 0.0001,
    sample_idx: int = 0,          
    num_samples: int = 100 
    ):
    if action == "check":
        print("="*60)
        print("Checking data files...")
        print("="*60)
        result = check_data_files.remote()
        print(f"\nResult: {result}")
        
        print("\n" + "="*60)
        print("Checking GPU setup...")
        print("="*60)
        gpu_info = check_gpu_setup.remote()
        print(f"\nGPU Info: {gpu_info}")
    
    elif action == "verify":
        print("="*60)
        print("Verifying compressed data...")
        print("="*60)
        result = decompress_data.remote()
        print(f"\nResult: {result}")
    
    elif action == "train":
        print("="*60)
        print("Starting model training...")
        print("="*60)
        result = train_model.remote(
            hidden_dim=hidden_dim,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        print(f"\nTraining Result: {result}")
    
    elif action == "inference":
        print("="*60)
        print(f"Running inference on sample {sample_idx}...")  
        print("="*60)
        result = run_inference.remote(sample_idx=sample_idx) 
        print(f"\nInference Result: {result}")
    
    elif action == "batch_inference":
        print("="*60)
        print(f"Running batch inference on {num_samples} samples...")
        print("="*60)
        result = batch_inference.remote(num_samples=num_samples)
        print(f"\nBatch Inference Result: {result}")
    
    elif action == "history":
        print("="*60)
        print("Getting training history...")
        print("="*60)
        result = get_training_history.remote()
        print(f"\nHistory: {result}")
    
    else:
        print(f"Unknown action: {action}")
        print("Available actions: check, verify, train, inference, history")


if __name__ == "__main__":
    pass