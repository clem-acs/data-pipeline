#!/usr/bin/env python3
"""
Script to analyze the distance data from channel_distances.csv
"""

import pandas as pd
import numpy as np
import os

# Load the CSV file
csv_path = os.path.join(os.path.dirname(__file__), 'channel_distances.csv')
df = pd.read_csv(csv_path)

print(f"Total channels: {len(df)}")

# Distance thresholds
print("\nDistance Thresholds:")
for threshold in [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]:
    count = len(df[df['distance'] < threshold])
    percentage = count / len(df) * 100
    print(f"Distances < {threshold}cm: {count} ({percentage:.2f}%)")

# Same module analysis
same_module = df[df['source_module'] == df['detector_module']]
print("\nSame Source and Detector Module:")
print(f"Count: {len(same_module)} ({len(same_module)/len(df)*100:.2f}%)")
print(f"Min distance: {same_module['distance'].min():.2f}cm")
print(f"Max distance: {same_module['distance'].max():.2f}cm")
print(f"Mean distance: {same_module['distance'].mean():.2f}cm")
print(f"Median distance: {same_module['distance'].median():.2f}cm")

# Different module analysis
diff_module = df[df['source_module'] != df['detector_module']]
print("\nDifferent Source and Detector Module:")
print(f"Count: {len(diff_module)} ({len(diff_module)/len(df)*100:.2f}%)")
print(f"Min distance: {diff_module['distance'].min():.2f}cm")
print(f"Max distance: {diff_module['distance'].max():.2f}cm")
print(f"Mean distance: {diff_module['distance'].mean():.2f}cm")
print(f"Median distance: {diff_module['distance'].median():.2f}cm")

# Source-detector ID analysis within same module
print("\nSame module, relationship between source ID and detector ID:")
same_module_stats = []
for source_id in range(1, 4):
    for detector_id in range(1, 7):
        subset = same_module[(same_module['source_id'] == source_id) & 
                             (same_module['detector_id'] == detector_id)]
        if len(subset) > 0:
            same_module_stats.append({
                'source_id': source_id,
                'detector_id': detector_id,
                'count': len(subset),
                'min_distance': subset['distance'].min(),
                'max_distance': subset['distance'].max(),
                'mean_distance': subset['distance'].mean()
            })

# Print in a tabular format
print("\nDistance by source_id and detector_id combinations (same module):")
print("source_id | detector_id | count | min_dist | max_dist | mean_dist")
print("-" * 65)
for stat in same_module_stats:
    print(f"{stat['source_id']:9} | {stat['detector_id']:11} | {stat['count']:5} | "
          f"{stat['min_distance']:8.2f} | {stat['max_distance']:8.2f} | {stat['mean_distance']:9.2f}")

# Find reasonably close distances for practical measurement
reasonable_distances = df[(df['distance'] >= 1.0) & (df['distance'] <= 5.0)]
print(f"\nChannels with reasonable distances (1-5cm): {len(reasonable_distances)}")

typical_ranges = [(0, 1), (1, 3), (3, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, float('inf'))]
print("\nDistance Range Distribution:")
for low, high in typical_ranges:
    if high == float('inf'):
        count = len(df[df['distance'] >= low])
        print(f"{low}+ cm: {count} ({count/len(df)*100:.2f}%)")
    else:
        count = len(df[(df['distance'] >= low) & (df['distance'] < high)])
        print(f"{low}-{high} cm: {count} ({count/len(df)*100:.2f}%)")