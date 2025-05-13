#!/usr/bin/env python3
"""
Timestamp Analysis Script for H5 Files

This script analyzes event timestamps in H5 files, focusing on:
1. Time differences between element_sent and element_replied events
2. Time differences between task_started and task_completed events
3. Identifying gaps between server and client timestamps for the same event
4. Detecting unexpected timing patterns

Usage:
    python time.py /path/to/h5file.h5
"""

import argparse
import h5py
import json
import numpy as np
import sys
import os
from collections import defaultdict
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Constants
CLIENT_TS_IDX = 0
SERVER_TS_IDX = 1
TIME_THRESHOLD_MS = 500  # 500ms threshold for "large" time differences

def load_h5_events(h5_path):
    """
    Load events from an H5 file
    
    Args:
        h5_path: Path to the H5 file
        
    Returns:
        Dictionary containing event data by event type
    """
    if not os.path.exists(h5_path):
        print(f"Error: File {h5_path} not found")
        sys.exit(1)
    
    print(f"Loading events from {h5_path}...")
    events = {}
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Check if events group exists
            if 'events' not in f:
                print("Error: No events found in the H5 file")
                sys.exit(1)
            
            events_group = f['events']
            
            # Process each event type
            for event_type in events_group.keys():
                event_group = events_group[event_type]
                
                # Each event type should have data, event_ids, and timestamps
                if all(key in event_group for key in ['data', 'event_ids', 'timestamps']):
                    events[event_type] = {
                        'data': [json.loads(data) for data in event_group['data'][:]], 
                        'event_ids': event_group['event_ids'][:],
                        'timestamps': event_group['timestamps'][:]
                    }
    except Exception as e:
        print(f"Error loading H5 file: {e}")
        sys.exit(1)
    
    return events

def parse_event_data(events):
    """
    Parse event data and organize by element ID and event type
    
    Args:
        events: Dictionary of raw events
        
    Returns:
        Tuple of (elements_by_id, events_by_type)
    """
    elements_by_id = defaultdict(list)
    events_by_type = defaultdict(list)
    
    # Process element_sent events
    if 'element_sent' in events:
        for i, (data, event_id, timestamps) in enumerate(zip(
            events['element_sent']['data'],
            events['element_sent']['event_ids'],
            events['element_sent']['timestamps']
        )):
            element_id = data.get('element_id', '')
            if element_id:
                elements_by_id[element_id].append({
                    'type': 'sent',
                    'data': data,
                    'event_id': event_id.decode('utf-8') if isinstance(event_id, bytes) else event_id,
                    'client_timestamp': timestamps[CLIENT_TS_IDX],
                    'server_timestamp': timestamps[SERVER_TS_IDX],
                    'index': i
                })
            
            # Also add to events by type
            events_by_type['element_sent'].append({
                'element_id': element_id,
                'data': data,
                'event_id': event_id.decode('utf-8') if isinstance(event_id, bytes) else event_id,
                'client_timestamp': timestamps[CLIENT_TS_IDX],
                'server_timestamp': timestamps[SERVER_TS_IDX],
                'index': i
            })
    
    # Process element_replied events
    if 'element_replied' in events:
        for i, (data, event_id, timestamps) in enumerate(zip(
            events['element_replied']['data'],
            events['element_replied']['event_ids'],
            events['element_replied']['timestamps']
        )):
            element_id = data.get('element_id', '')
            if element_id:
                elements_by_id[element_id].append({
                    'type': 'replied',
                    'data': data,
                    'event_id': event_id.decode('utf-8') if isinstance(event_id, bytes) else event_id,
                    'client_timestamp': timestamps[CLIENT_TS_IDX],
                    'server_timestamp': timestamps[SERVER_TS_IDX],
                    'index': i
                })
            
            # Also add to events by type
            events_by_type['element_replied'].append({
                'element_id': element_id,
                'data': data,
                'event_id': event_id.decode('utf-8') if isinstance(event_id, bytes) else event_id,
                'client_timestamp': timestamps[CLIENT_TS_IDX],
                'server_timestamp': timestamps[SERVER_TS_IDX],
                'index': i
            })
    
    # Process task events
    for event_type in ['task_started', 'task_completed']:
        if event_type in events:
            for i, (data, event_id, timestamps) in enumerate(zip(
                events[event_type]['data'],
                events[event_type]['event_ids'],
                events[event_type]['timestamps']
            )):
                task_id = data.get('task_id', '')
                events_by_type[event_type].append({
                    'task_id': task_id,
                    'data': data,
                    'event_id': event_id.decode('utf-8') if isinstance(event_id, bytes) else event_id,
                    'client_timestamp': timestamps[CLIENT_TS_IDX],
                    'server_timestamp': timestamps[SERVER_TS_IDX],
                    'index': i
                })
    
    # Process other event types
    for event_type in events:
        if event_type not in ['element_sent', 'element_replied', 'task_started', 'task_completed']:
            for i, (data, event_id, timestamps) in enumerate(zip(
                events[event_type]['data'],
                events[event_type]['event_ids'],
                events[event_type]['timestamps']
            )):
                events_by_type[event_type].append({
                    'data': data,
                    'event_id': event_id.decode('utf-8') if isinstance(event_id, bytes) else event_id,
                    'client_timestamp': timestamps[CLIENT_TS_IDX],
                    'server_timestamp': timestamps[SERVER_TS_IDX],
                    'index': i
                })
    
    return elements_by_id, events_by_type

def analyze_element_durations(elements_by_id):
    """
    Analyze time differences between element_sent and element_replied events
    
    Args:
        elements_by_id: Dictionary mapping element IDs to event lists
        
    Returns:
        List of duration records
    """
    durations = []
    
    for element_id, events in elements_by_id.items():
        sent_events = [e for e in events if e['type'] == 'sent']
        replied_events = [e for e in events if e['type'] == 'replied']
        
        # Skip if we don't have both sent and replied events
        if not sent_events or not replied_events:
            continue
        
        # Handle multiple sent/replied events for the same element ID
        for replied in replied_events:
            # Find the matching sent event (closest before the reply)
            sent_candidates = [s for s in sent_events if s['client_timestamp'] < replied['client_timestamp']]
            
            if sent_candidates:
                # Get the most recent sent event before the reply
                sent = max(sent_candidates, key=lambda s: s['client_timestamp'])
                
                # Calculate durations
                client_duration_ms = replied['client_timestamp'] - sent['client_timestamp']
                server_duration_ms = replied['server_timestamp'] - sent['server_timestamp']
                
                # Record the duration
                durations.append({
                    'element_id': element_id,
                    'sent_event_id': sent['event_id'],
                    'replied_event_id': replied['event_id'],
                    'client_sent_ts': sent['client_timestamp'],
                    'client_replied_ts': replied['client_timestamp'],
                    'server_sent_ts': sent['server_timestamp'],
                    'server_replied_ts': replied['server_timestamp'],
                    'client_duration_ms': client_duration_ms,
                    'server_duration_ms': server_duration_ms,
                    'ts_difference_ms': abs(client_duration_ms - server_duration_ms)
                })
    
    return durations

def analyze_task_durations(events_by_type):
    """
    Analyze time differences between task_started and task_completed events
    
    Args:
        events_by_type: Dictionary mapping event types to event lists
        
    Returns:
        List of task duration records
    """
    task_durations = []
    
    # Check if we have both task_started and task_completed events
    if 'task_started' not in events_by_type or 'task_completed' not in events_by_type:
        return task_durations
    
    # Group events by task_id
    started_by_task = {e['task_id']: e for e in events_by_type['task_started'] if e['task_id']}
    completed_by_task = {e['task_id']: e for e in events_by_type['task_completed'] if e['task_id']}
    
    # Find matching tasks
    for task_id in set(started_by_task.keys()) & set(completed_by_task.keys()):
        started = started_by_task[task_id]
        completed = completed_by_task[task_id]
        
        # Calculate durations
        client_duration_ms = completed['client_timestamp'] - started['client_timestamp']
        server_duration_ms = completed['server_timestamp'] - started['server_timestamp']
        
        # Record the duration
        task_durations.append({
            'task_id': task_id,
            'task_type': started['data'].get('task_type', ''),
            'started_event_id': started['event_id'],
            'completed_event_id': completed['event_id'],
            'client_started_ts': started['client_timestamp'],
            'client_completed_ts': completed['client_timestamp'],
            'server_started_ts': started['server_timestamp'],
            'server_completed_ts': completed['server_timestamp'],
            'client_duration_ms': client_duration_ms,
            'server_duration_ms': server_duration_ms,
            'ts_difference_ms': abs(client_duration_ms - server_duration_ms)
        })
    
    return task_durations

def analyze_timestamp_gaps(events_by_type):
    """
    Analyze gaps between client and server timestamps for the same events

    Args:
        events_by_type: Dictionary mapping event types to event lists

    Returns:
        Dictionary with timestamp gap analysis results
    """
    results = {
        'large_gaps': [],
        'gap_stats_by_type': {}
    }

    for event_type, events in events_by_type.items():
        gaps = []

        for event in events:
            # Calculate the absolute gap between client and server timestamps
            gap_ms = abs(event['client_timestamp'] - event['server_timestamp'])
            gaps.append(gap_ms)

            # Record large gaps
            if gap_ms > TIME_THRESHOLD_MS:
                results['large_gaps'].append({
                    'event_type': event_type,
                    'event_id': event['event_id'],
                    'client_timestamp': event['client_timestamp'],
                    'server_timestamp': event['server_timestamp'],
                    'gap_ms': gap_ms
                })

        # Calculate gap statistics for this event type
        if gaps:
            results['gap_stats_by_type'][event_type] = {
                'min_ms': min(gaps),
                'max_ms': max(gaps),
                'mean_ms': np.mean(gaps),
                'median_ms': np.median(gaps),
                'std_ms': np.std(gaps),
                'event_count': len(gaps)
            }

    return results

def analyze_inter_event_timing(events_by_type):
    """
    Analyze timing patterns between consecutive events of the same type
    
    Args:
        events_by_type: Dictionary mapping event types to event lists
        
    Returns:
        Dictionary with timing analysis results
    """
    results = {
        'large_gaps': [],
        'timing_stats_by_type': {}
    }
    
    for event_type, events in events_by_type.items():
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e['client_timestamp'])
        
        if len(sorted_events) < 2:
            continue
        
        # Calculate time differences between consecutive events
        client_diffs = []
        server_diffs = []
        
        for i in range(1, len(sorted_events)):
            prev = sorted_events[i-1]
            curr = sorted_events[i]
            
            client_diff = curr['client_timestamp'] - prev['client_timestamp']
            server_diff = curr['server_timestamp'] - prev['server_timestamp']
            
            client_diffs.append(client_diff)
            server_diffs.append(server_diff)
            
            # Identify abnormally large gaps (beyond 3 std dev for this event type)
            if i > 1 and len(client_diffs) > 3:  # Need at least a few samples to establish a pattern
                mean_diff = np.mean(client_diffs[:-1])
                std_diff = np.std(client_diffs[:-1])
                if std_diff > 0 and client_diff > mean_diff + 3*std_diff:
                    results['large_gaps'].append({
                        'event_type': event_type,
                        'prev_event_id': prev['event_id'],
                        'curr_event_id': curr['event_id'],
                        'client_gap_ms': client_diff,
                        'server_gap_ms': server_diff,
                        'mean_gap_ms': mean_diff,
                        'std_dev_ms': std_diff,
                        'z_score': (client_diff - mean_diff) / std_diff if std_diff > 0 else 0
                    })
        
        # Calculate timing statistics
        if client_diffs:
            results['timing_stats_by_type'][event_type] = {
                'client': {
                    'min_ms': min(client_diffs),
                    'max_ms': max(client_diffs),
                    'mean_ms': np.mean(client_diffs),
                    'median_ms': np.median(client_diffs),
                    'std_ms': np.std(client_diffs)
                },
                'server': {
                    'min_ms': min(server_diffs),
                    'max_ms': max(server_diffs),
                    'mean_ms': np.mean(server_diffs),
                    'median_ms': np.median(server_diffs),
                    'std_ms': np.std(server_diffs)
                },
                'event_count': len(sorted_events)
            }
    
    return results

def format_timestamp(ts_ms):
    """Convert millisecond timestamp to readable format"""
    dt = datetime.fromtimestamp(ts_ms / 1000)
    return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

def print_analysis_results(durations, task_durations, gap_analysis, timing_analysis):
    """
    Print formatted analysis results
    
    Args:
        durations: List of element duration records
        task_durations: List of task duration records
        gap_analysis: Results from timestamp gap analysis
        timing_analysis: Results from inter-event timing analysis
    """
    print("\n" + "="*80)
    print("TIMESTAMP ANALYSIS RESULTS")
    print("="*80)
    
    # 1. Element Send/Reply Duration Analysis
    print("\n1. ELEMENT SEND/REPLY DURATIONS")
    print("-" * 40)
    
    if durations:
        # Create DataFrame for easier manipulation
        df = pd.DataFrame(durations)
        
        # Display basic statistics
        print(f"Total elements analyzed: {len(df)}")
        print("\nDuration statistics (milliseconds):")
        print(f"  Client-side durations:")
        print(f"    Min: {df['client_duration_ms'].min():.2f} ms")
        print(f"    Max: {df['client_duration_ms'].max():.2f} ms")
        print(f"    Mean: {df['client_duration_ms'].mean():.2f} ms")
        print(f"    Median: {df['client_duration_ms'].median():.2f} ms")
        print(f"    Std Dev: {df['client_duration_ms'].std():.2f} ms")
        
        print(f"\n  Server-side durations:")
        print(f"    Min: {df['server_duration_ms'].min():.2f} ms")
        print(f"    Max: {df['server_duration_ms'].max():.2f} ms")
        print(f"    Mean: {df['server_duration_ms'].mean():.2f} ms")
        print(f"    Median: {df['server_duration_ms'].median():.2f} ms")
        print(f"    Std Dev: {df['server_duration_ms'].std():.2f} ms")
        
        # Display individual durations
        print("\nIndividual element durations:")
        for i, row in df.iterrows():
            print(f"\nElement {i+1}: {row['element_id']}")
            print(f"  Sent event: {row['sent_event_id']}")
            print(f"  Replied event: {row['replied_event_id']}")
            print(f"  Client timestamps: {format_timestamp(row['client_sent_ts'])} -> {format_timestamp(row['client_replied_ts'])}")
            print(f"  Server timestamps: {format_timestamp(row['server_sent_ts'])} -> {format_timestamp(row['server_replied_ts'])}")
            print(f"  Client duration: {row['client_duration_ms']:.2f} ms")
            print(f"  Server duration: {row['server_duration_ms']:.2f} ms")
            print(f"  Client/Server difference: {row['ts_difference_ms']:.2f} ms")
    else:
        print("No element send/reply pairs found for analysis.")
    
    # 2. Task Start/Complete Duration Analysis
    print("\n\n2. TASK START/COMPLETE DURATIONS")
    print("-" * 40)
    
    if task_durations:
        # Create DataFrame for easier manipulation
        task_df = pd.DataFrame(task_durations)
        
        # Display basic statistics
        print(f"Total tasks analyzed: {len(task_df)}")
        print("\nDuration statistics (milliseconds):")
        print(f"  Client-side durations:")
        print(f"    Min: {task_df['client_duration_ms'].min():.2f} ms")
        print(f"    Max: {task_df['client_duration_ms'].max():.2f} ms")
        print(f"    Mean: {task_df['client_duration_ms'].mean():.2f} ms")
        print(f"    Median: {task_df['client_duration_ms'].median():.2f} ms")
        print(f"    Std Dev: {task_df['client_duration_ms'].std():.2f} ms")
        
        print(f"\n  Server-side durations:")
        print(f"    Min: {task_df['server_duration_ms'].min():.2f} ms")
        print(f"    Max: {task_df['server_duration_ms'].max():.2f} ms")
        print(f"    Mean: {task_df['server_duration_ms'].mean():.2f} ms")
        print(f"    Median: {task_df['server_duration_ms'].median():.2f} ms")
        print(f"    Std Dev: {task_df['server_duration_ms'].std():.2f} ms")
        
        # Display individual task durations
        print("\nIndividual task durations:")
        for i, row in task_df.iterrows():
            print(f"\nTask {i+1}: {row['task_id']} (Type: {row['task_type']})")
            print(f"  Started event: {row['started_event_id']}")
            print(f"  Completed event: {row['completed_event_id']}")
            print(f"  Client timestamps: {format_timestamp(row['client_started_ts'])} -> {format_timestamp(row['client_completed_ts'])}")
            print(f"  Server timestamps: {format_timestamp(row['server_started_ts'])} -> {format_timestamp(row['server_completed_ts'])}")
            print(f"  Client duration: {row['client_duration_ms']:.2f} ms")
            print(f"  Server duration: {row['server_duration_ms']:.2f} ms")
            print(f"  Client/Server difference: {row['ts_difference_ms']:.2f} ms")
    else:
        print("No task start/complete pairs found for analysis.")
    
    # 3. Client/Server Timestamp Gap Analysis
    print("\n\n3. CLIENT/SERVER TIMESTAMP GAPS")
    print("-" * 40)
    
    # Display gap statistics by event type
    print("Gap statistics by event type (milliseconds):")
    for event_type, stats in gap_analysis['gap_stats_by_type'].items():
        print(f"\n  {event_type} ({stats['event_count']} events):")
        print(f"    Min: {stats['min_ms']:.2f} ms")
        print(f"    Max: {stats['max_ms']:.2f} ms")
        print(f"    Mean: {stats['mean_ms']:.2f} ms")
        print(f"    Median: {stats['median_ms']:.2f} ms")
        print(f"    Std Dev: {stats['std_ms']:.2f} ms")
    
    # Display large gaps
    if gap_analysis['large_gaps']:
        print(f"\nLarge gaps between client and server timestamps (>{TIME_THRESHOLD_MS}ms):")
        for i, gap in enumerate(sorted(gap_analysis['large_gaps'], key=lambda g: g['gap_ms'], reverse=True)):
            print(f"\n  Gap {i+1}: {gap['event_type']} - {gap['event_id']}")
            print(f"    Client timestamp: {format_timestamp(gap['client_timestamp'])}")
            print(f"    Server timestamp: {format_timestamp(gap['server_timestamp'])}")
            print(f"    Gap: {gap['gap_ms']:.2f} ms")
    else:
        print(f"\nNo large gaps (>{TIME_THRESHOLD_MS}ms) found between client and server timestamps.")
    
    # 4. Inter-event Timing Analysis
    print("\n\n4. INTER-EVENT TIMING ANALYSIS")
    print("-" * 40)
    
    # Display timing statistics by event type
    print("Timing statistics between consecutive events (milliseconds):")
    for event_type, stats in timing_analysis['timing_stats_by_type'].items():
        print(f"\n  {event_type} ({stats['event_count']} events):")
        print(f"    Client-side intervals:")
        print(f"      Min: {stats['client']['min_ms']:.2f} ms")
        print(f"      Max: {stats['client']['max_ms']:.2f} ms")
        print(f"      Mean: {stats['client']['mean_ms']:.2f} ms")
        print(f"      Median: {stats['client']['median_ms']:.2f} ms")
        print(f"      Std Dev: {stats['client']['std_ms']:.2f} ms")
        
        print(f"    Server-side intervals:")
        print(f"      Min: {stats['server']['min_ms']:.2f} ms")
        print(f"      Max: {stats['server']['max_ms']:.2f} ms")
        print(f"      Mean: {stats['server']['mean_ms']:.2f} ms")
        print(f"      Median: {stats['server']['median_ms']:.2f} ms")
        print(f"      Std Dev: {stats['server']['std_ms']:.2f} ms")
    
    # Display abnormally large gaps between consecutive events
    if timing_analysis['large_gaps']:
        print(f"\nAbnormally large gaps between consecutive events (>3σ):")
        for i, gap in enumerate(sorted(timing_analysis['large_gaps'], key=lambda g: g['z_score'], reverse=True)):
            print(f"\n  Gap {i+1}: {gap['event_type']}")
            print(f"    Between events: {gap['prev_event_id']} -> {gap['curr_event_id']}")
            print(f"    Client gap: {gap['client_gap_ms']:.2f} ms")
            print(f"    Server gap: {gap['server_gap_ms']:.2f} ms")
            print(f"    Mean gap for this event type: {gap['mean_gap_ms']:.2f} ms")
            print(f"    Standard deviation: {gap['std_dev_ms']:.2f} ms")
            print(f"    Z-score: {gap['z_score']:.2f}")
    else:
        print("\nNo abnormally large gaps found between consecutive events.")

def generate_plots(durations, task_durations, gap_analysis, timing_analysis, output_dir):
    """
    Generate visualization plots for the analysis results
    
    Args:
        durations: List of element duration records
        task_durations: List of task duration records
        gap_analysis: Results from timestamp gap analysis
        timing_analysis: Results from inter-event timing analysis
        output_dir: Directory to save the plots
    
    Returns:
        List of generated plot file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_files = []
    
    # 1. Element durations plot
    if durations:
        df = pd.DataFrame(durations)
        
        # Plot comparing client vs server durations
        plt.figure(figsize=(10, 6))
        plt.scatter(df['client_duration_ms'], df['server_duration_ms'], alpha=0.7)
        plt.plot([df['client_duration_ms'].min(), df['client_duration_ms'].max()], 
                [df['client_duration_ms'].min(), df['client_duration_ms'].max()], 
                'r--', alpha=0.5)
        plt.xlabel('Client Duration (ms)')
        plt.ylabel('Server Duration (ms)')
        plt.title('Element Send/Reply Durations: Client vs Server')
        plt.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = df['client_duration_ms'].corr(df['server_duration_ms'])
        plt.annotate(f'Correlation: {corr:.4f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=10, ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
        
        file_path = os.path.join(output_dir, 'element_durations.png')
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_files.append(file_path)
        
        # Histogram of timestamp differences
        plt.figure(figsize=(10, 6))
        plt.hist(df['ts_difference_ms'], bins=30, alpha=0.7)
        plt.xlabel('Client/Server Timestamp Difference (ms)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Client/Server Timestamp Differences for Elements')
        plt.grid(True, alpha=0.3)
        
        file_path = os.path.join(output_dir, 'element_timestamp_diffs.png')
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_files.append(file_path)
    
    # 2. Task durations plot
    if task_durations:
        task_df = pd.DataFrame(task_durations)
        
        # Bar plot of task durations by task_id
        plt.figure(figsize=(12, 8))
        tasks = task_df['task_id'].tolist()
        client_durations = task_df['client_duration_ms'].tolist()
        server_durations = task_df['server_duration_ms'].tolist()
        
        x = range(len(tasks))
        bar_width = 0.35
        
        plt.bar([i - bar_width/2 for i in x], client_durations, bar_width, alpha=0.7, label='Client')
        plt.bar([i + bar_width/2 for i in x], server_durations, bar_width, alpha=0.7, label='Server')
        
        plt.xlabel('Task ID')
        plt.ylabel('Duration (ms)')
        plt.title('Task Durations by Task ID')
        plt.xticks(x, tasks, rotation=90)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        file_path = os.path.join(output_dir, 'task_durations.png')
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_files.append(file_path)
    
    # 3. Client/Server timestamp gaps by event type
    if gap_analysis['gap_stats_by_type']:
        event_types = list(gap_analysis['gap_stats_by_type'].keys())
        means = [gap_analysis['gap_stats_by_type'][et]['mean_ms'] for et in event_types]
        maxes = [gap_analysis['gap_stats_by_type'][et]['max_ms'] for et in event_types]
        
        plt.figure(figsize=(12, 8))
        x = range(len(event_types))
        bar_width = 0.35
        
        plt.bar([i - bar_width/2 for i in x], means, bar_width, alpha=0.7, label='Mean Gap')
        plt.bar([i + bar_width/2 for i in x], maxes, bar_width, alpha=0.7, label='Max Gap')
        
        plt.xlabel('Event Type')
        plt.ylabel('Gap (ms)')
        plt.title('Client/Server Timestamp Gaps by Event Type')
        plt.xticks(x, event_types, rotation=90)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        file_path = os.path.join(output_dir, 'timestamp_gaps.png')
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_files.append(file_path)
    
    return plot_files

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Analyze timestamps in H5 files")
    parser.add_argument("h5_file", help="Path to the H5 file to analyze")
    parser.add_argument("--output-dir", "-o", help="Directory to save output files and plots")
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(args.h5_file) or '.'
    
    # Load events from the H5 file
    events = load_h5_events(args.h5_file)
    
    # Parse and organize events
    elements_by_id, events_by_type = parse_event_data(events)
    
    # Analyze element durations (element_sent → element_replied)
    durations = analyze_element_durations(elements_by_id)
    
    # Analyze task durations (task_started → task_completed)
    task_durations = analyze_task_durations(events_by_type)
    
    # Analyze timestamp gaps between client and server
    gap_analysis = analyze_timestamp_gaps(events_by_type)
    
    # Analyze timing patterns between consecutive events
    timing_analysis = analyze_inter_event_timing(events_by_type)
    
    # Print results
    print_analysis_results(durations, task_durations, gap_analysis, timing_analysis)
    
    # Generate plots
    plot_files = generate_plots(durations, task_durations, gap_analysis, timing_analysis, output_dir)
    
    if plot_files:
        print(f"\nGenerated {len(plot_files)} plots in {output_dir}")
        for plot_file in plot_files:
            print(f"  - {os.path.basename(plot_file)}")
    
    # Save detailed results to CSV files
    if durations:
        element_df = pd.DataFrame(durations)
        element_csv = os.path.join(output_dir, f"{Path(args.h5_file).stem}_element_durations.csv")
        element_df.to_csv(element_csv, index=False)
        print(f"\nSaved element durations to {element_csv}")
    
    if task_durations:
        task_df = pd.DataFrame(task_durations)
        task_csv = os.path.join(output_dir, f"{Path(args.h5_file).stem}_task_durations.csv")
        task_df.to_csv(task_csv, index=False)
        print(f"Saved task durations to {task_csv}")
    
    # Save gap analysis
    if gap_analysis['large_gaps']:
        gap_df = pd.DataFrame(gap_analysis['large_gaps'])
        gap_csv = os.path.join(output_dir, f"{Path(args.h5_file).stem}_timestamp_gaps.csv")
        gap_df.to_csv(gap_csv, index=False)
        print(f"Saved timestamp gaps to {gap_csv}")
    
    # Save timing analysis
    if timing_analysis['large_gaps']:
        timing_df = pd.DataFrame(timing_analysis['large_gaps'])
        timing_csv = os.path.join(output_dir, f"{Path(args.h5_file).stem}_timing_gaps.csv")
        timing_df.to_csv(timing_csv, index=False)
        print(f"Saved timing gaps to {timing_csv}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()