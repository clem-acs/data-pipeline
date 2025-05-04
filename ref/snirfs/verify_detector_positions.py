#!/usr/bin/env python3
"""
Script to verify detector positions within modules in layout.json
to determine if position 7 is consistently in the center.
"""
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def load_layout_json(file_path):
    """Load detector positions from layout.json."""
    with open(file_path, 'r') as f:
        layout_data = json.load(f)
    
    # Extract detector positions
    detector_modules = []
    for module_idx, module in enumerate(layout_data.get("detector_locations", [])):
        module_detectors = []
        for pos_idx, pos in enumerate(module):
            module_detectors.append({
                "module_idx": module_idx + 1,
                "pos_idx": pos_idx + 1,
                "global_idx": len([d for m in detector_modules for d in m]) + pos_idx + 1,
                "pos": pos
            })
        detector_modules.append(module_detectors)
    
    return detector_modules

def calculate_centroid(positions):
    """Calculate the centroid of a set of positions."""
    return np.mean(positions, axis=0)

def calculate_distances_from_centroid(positions, centroid):
    """Calculate distances of each position from the centroid."""
    return [np.sqrt(np.sum((np.array(pos) - np.array(centroid))**2)) for pos in positions]

def analyze_module_geometry(modules):
    """Analyze the geometric arrangement of detectors within each module."""
    module_analysis = []
    
    for module_idx, module in enumerate(modules):
        # Extract positions and calculate centroid
        positions = [detector["pos"] for detector in module]
        centroid = calculate_centroid(positions)
        
        # Calculate distances from centroid
        distances = []
        for i, detector in enumerate(module):
            pos = np.array(detector["pos"])
            distance = np.sqrt(np.sum((pos - centroid)**2))
            distances.append({
                "pos_idx": detector["pos_idx"],
                "distance": distance
            })
        
        # Sort by distance from centroid
        sorted_distances = sorted(distances, key=lambda x: x["distance"])
        
        # Find if position 7 is among the N closest to centroid
        pos7_rank = next((i+1 for i, d in enumerate(sorted_distances) if d["pos_idx"] == 7), None)
        
        # Store module analysis
        module_analysis.append({
            "module_idx": module_idx + 1,
            "centroid": centroid.tolist(),
            "sorted_positions": [d["pos_idx"] for d in sorted_distances],
            "position_distances": {d["pos_idx"]: d["distance"] for d in distances},
            "pos7_rank": pos7_rank,
            "pos7_distance": next((d["distance"] for d in distances if d["pos_idx"] == 7), None)
        })
    
    return module_analysis

def visualize_module(module, module_idx, analysis, output_dir):
    """Create a 2D visualization of detector positions in a module."""
    plt.figure(figsize=(8, 6))
    
    # Extract positions for plotting
    positions = np.array([detector["pos"] for detector in module])
    centroid = np.array(analysis["centroid"])
    
    # Project 3D coordinates to 2D for visualization
    # Use first two coordinates (X, Y) as the main axes
    x = positions[:, 0]
    y = positions[:, 1]
    
    # Plot all positions
    plt.scatter(x, y, s=100, c='blue', marker='o', label='Detector')
    
    # Plot position indices
    for i, detector in enumerate(module):
        plt.text(x[i], y[i], str(detector["pos_idx"]), 
                fontsize=12, ha='center', va='center')
    
    # Plot centroid
    plt.scatter(centroid[0], centroid[1], s=150, c='red', marker='*', label='Centroid')
    
    # Add circles representing distance from centroid
    for radius in [10, 20, 30]:
        circle = Circle((centroid[0], centroid[1]), radius, 
                        fill=False, linestyle='--', alpha=0.3)
        plt.gca().add_patch(circle)
    
    # Set plot title and labels
    plt.title(f"Module {module_idx} Detector Positions")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_dir / f"module_{module_idx}_positions.png")
    plt.close()

def generate_report(analysis_data, output_dir):
    """Generate a report of detector positions analysis."""
    report = "# Detector Positions Analysis\n\n"
    report += "This report analyzes the geometric arrangement of detectors within each module.\n\n"
    
    # Summarize position 7 placement across modules
    pos7_central_count = sum(1 for m in analysis_data if m["pos7_rank"] == 1)
    pos7_near_central_count = sum(1 for m in analysis_data if m["pos7_rank"] in [1, 2, 3])
    
    report += "## Position 7 Placement Summary\n\n"
    report += f"- Modules where position 7 is the closest to centroid: {pos7_central_count}/{len(analysis_data)}\n"
    report += f"- Modules where position 7 is among the 3 closest to centroid: {pos7_near_central_count}/{len(analysis_data)}\n\n"
    
    # Add a table of position ranks by closeness to centroid
    report += "## Position Rankings by Closeness to Centroid\n\n"
    report += "This table shows for each module which positions are closest to the geometric center (centroid):\n\n"
    report += "| Module | 1st (closest) | 2nd | 3rd | 4th | 5th | 6th | 7th (farthest) |\n"
    report += "|--------|--------------|-----|-----|-----|-----|-----|----------------|\n"
    
    for module in analysis_data:
        ranks = module["sorted_positions"]
        report += f"| {module['module_idx']} | {ranks[0]} | {ranks[1]} | {ranks[2]} | {ranks[3]} | {ranks[4]} | {ranks[5]} | {ranks[6]} |\n"
    
    # Add detailed analysis of each module
    report += "\n## Detailed Module Analysis\n\n"
    
    for module in analysis_data:
        report += f"### Module {module['module_idx']}\n\n"
        report += f"- Position 7 rank: {module['pos7_rank']} of 7\n"
        report += f"- Position 7 distance from centroid: {module['pos7_distance']:.2f} units\n"
        report += "- Distance of each position from centroid:\n\n"
        
        # Create a table of distances
        report += "| Position | Distance from Center |\n"
        report += "|----------|----------------------|\n"
        
        for pos in range(1, 8):
            distance = module["position_distances"].get(pos, "N/A")
            if distance != "N/A":
                distance = f"{distance:.2f}"
            report += f"| {pos} | {distance} |\n"
        
        report += "\n"
    
    # Write report to file
    report_path = output_dir / "detector_positions_analysis.md"
    with open(report_path, "w") as f:
        f.write(report)
    
    return report_path

def main():
    """Main function to verify detector positions."""
    layout_path = "/Users/clem/Desktop/code/data-pipeline/ref/layout.json"
    output_dir = Path("/Users/clem/Desktop/code/data-pipeline/ref/snirfs")
    
    # Load layout.json
    detector_modules = load_layout_json(layout_path)
    print(f"Loaded {len(detector_modules)} detector modules from layout.json")
    
    # Analyze module geometry
    module_analysis = analyze_module_geometry(detector_modules)
    
    # Generate visualizations for each module
    viz_dir = output_dir / "module_visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    print("Generating module visualizations...")
    for module_idx, module in enumerate(detector_modules):
        visualize_module(module, module_idx+1, module_analysis[module_idx], viz_dir)
    
    # Generate report
    report_path = generate_report(module_analysis, output_dir)
    print(f"Analysis report written to: {report_path}")
    
    # Save raw analysis data
    json_path = output_dir / "detector_positions_analysis.json"
    with open(json_path, "w") as f:
        json.dump(module_analysis, f, indent=2)
    
    print(f"Raw analysis data saved to: {json_path}")

if __name__ == "__main__":
    main()