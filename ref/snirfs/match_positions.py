#!/usr/bin/env python3
"""
Script to identify which subset of layout.json positions correspond to SNIRF file positions,
using position-based matching instead of indices.
"""
import os
import json
import numpy as np
import h5py
from pathlib import Path
import argparse
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def extract_snirf_positions(file_path):
    """Extract source and detector positions from a SNIRF file."""
    try:
        results = {
            "file_name": os.path.basename(file_path),
            "sources": [],
            "detectors": [],
            "error": None
        }
        
        with h5py.File(file_path, 'r') as snirf_file:
            if 'nirs' not in snirf_file:
                results["error"] = "No 'nirs' group found"
                return results
            
            nirs = snirf_file['nirs']
            
            if 'probe' not in nirs:
                results["error"] = "No 'probe' group found"
                return results
            
            probe = nirs['probe']
            
            # Extract source positions
            if 'sourcePos3D' in probe:
                sources = probe['sourcePos3D'][:]
                for i, pos in enumerate(sources):
                    results["sources"].append({
                        "index": i + 1,
                        "pos": [float(pos[0]), float(pos[1]), float(pos[2])]
                    })
            elif 'sourcePos' in probe:
                sources = probe['sourcePos'][:]
                for i, pos in enumerate(sources):
                    results["sources"].append({
                        "index": i + 1,
                        "pos": [float(pos[0]), float(pos[1]), float(pos[2]) if len(pos) > 2 else 0.0]
                    })
            
            # Extract detector positions
            if 'detectorPos3D' in probe:
                detectors = probe['detectorPos3D'][:]
                for i, pos in enumerate(detectors):
                    results["detectors"].append({
                        "index": i + 1,
                        "pos": [float(pos[0]), float(pos[1]), float(pos[2])]
                    })
            elif 'detectorPos' in probe:
                detectors = probe['detectorPos'][:]
                for i, pos in enumerate(detectors):
                    results["detectors"].append({
                        "index": i + 1,
                        "pos": [float(pos[0]), float(pos[1]), float(pos[2]) if len(pos) > 2 else 0.0]
                    })
            
        return results
    
    except Exception as e:
        return {
            "file_name": os.path.basename(file_path),
            "error": str(e),
            "sources": [],
            "detectors": []
        }

def load_layout_json(file_path):
    """Load source and detector positions from layout.json."""
    with open(file_path, 'r') as f:
        layout_data = json.load(f)
    
    # Extract source positions as a flat list
    source_positions = []
    for group_idx, group in enumerate(layout_data.get("source_locations", [])):
        for i, pos in enumerate(group):
            source_positions.append({
                "layout_index": len(source_positions) + 1,
                "group": group_idx + 1,
                "pos_in_group": i + 1,
                "pos": pos
            })
    
    # Extract detector positions
    detector_positions = []
    for group_idx, group in enumerate(layout_data.get("detector_locations", [])):
        for i, pos in enumerate(group):
            detector_positions.append({
                "layout_index": len(detector_positions) + 1,
                "group": group_idx + 1,
                "pos_in_group": i + 1,
                "pos": pos
            })
    
    return {
        "sources": source_positions,
        "detectors": detector_positions
    }

def try_coordinate_transforms(snirf_positions, layout_positions):
    """
    Try various coordinate transformations to find the best match between 
    SNIRF and layout positions.
    """
    snirf_pos_array = np.array([s["pos"] for s in snirf_positions])
    layout_pos_array = np.array([l["pos"] for l in layout_positions])
    
    # Possible transformations to try
    transforms = [
        {"name": "Identity", "func": lambda x: x},
        {"name": "Negate X", "func": lambda x: np.column_stack([-x[:,0], x[:,1], x[:,2]])},
        {"name": "Negate Y", "func": lambda x: np.column_stack([x[:,0], -x[:,1], x[:,2]])},
        {"name": "Negate Z", "func": lambda x: np.column_stack([x[:,0], x[:,1], -x[:,2]])},
        {"name": "Negate X,Y", "func": lambda x: np.column_stack([-x[:,0], -x[:,1], x[:,2]])},
        {"name": "Negate X,Z", "func": lambda x: np.column_stack([-x[:,0], x[:,1], -x[:,2]])},
        {"name": "Negate Y,Z", "func": lambda x: np.column_stack([x[:,0], -x[:,1], -x[:,2]])},
        {"name": "Negate X,Y,Z", "func": lambda x: np.column_stack([-x[:,0], -x[:,1], -x[:,2]])},
        # Swap axes
        {"name": "Swap X,Y", "func": lambda x: np.column_stack([x[:,1], x[:,0], x[:,2]])},
        {"name": "Swap X,Z", "func": lambda x: np.column_stack([x[:,2], x[:,1], x[:,0]])},
        {"name": "Swap Y,Z", "func": lambda x: np.column_stack([x[:,0], x[:,2], x[:,1]])},
        # Swap and negate
        {"name": "Swap X,Y Negate X", "func": lambda x: np.column_stack([-x[:,1], x[:,0], x[:,2]])},
        {"name": "Swap X,Y Negate Y", "func": lambda x: np.column_stack([x[:,1], -x[:,0], x[:,2]])},
        {"name": "Swap X,Z Negate X", "func": lambda x: np.column_stack([-x[:,2], x[:,1], x[:,0]])},
        {"name": "Swap X,Z Negate Z", "func": lambda x: np.column_stack([x[:,2], x[:,1], -x[:,0]])},
    ]
    
    best_transform = None
    best_cost = float('inf')
    best_assignments = None
    
    for transform in transforms:
        transformed_snirf = transform["func"](snirf_pos_array)
        
        # Calculate distance matrix
        dist_matrix = cdist(transformed_snirf, layout_pos_array)
        
        # Use Hungarian algorithm to find optimal assignment
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        total_cost = dist_matrix[row_ind, col_ind].sum()
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_transform = transform
            best_assignments = (row_ind, col_ind)
    
    return best_transform, best_cost, best_assignments

def match_positions(snirf_data, layout_data):
    """
    Match SNIRF positions to layout.json positions based on coordinates,
    trying different transformations to find the best match.
    """
    # First find the best transform for sources
    best_source_transform, source_cost, source_assignments = try_coordinate_transforms(
        snirf_data["sources"], layout_data["sources"]
    )
    
    # Then find the best transform for detectors
    best_detector_transform, detector_cost, detector_assignments = try_coordinate_transforms(
        snirf_data["detectors"], layout_data["detectors"]
    )
    
    # Calculate average position error after transformation
    snirf_source_array = np.array([s["pos"] for s in snirf_data["sources"]])
    layout_source_array = np.array([l["pos"] for l in layout_data["sources"]])
    transformed_sources = best_source_transform["func"](snirf_source_array)
    
    snirf_detector_array = np.array([d["pos"] for d in snirf_data["detectors"]])
    layout_detector_array = np.array([l["pos"] for l in layout_data["detectors"]])
    transformed_detectors = best_detector_transform["func"](snirf_detector_array)
    
    row_ind, col_ind = source_assignments
    source_errors = np.sqrt(np.sum((transformed_sources[row_ind] - layout_source_array[col_ind])**2, axis=1))
    
    row_ind, col_ind = detector_assignments
    detector_errors = np.sqrt(np.sum((transformed_detectors[row_ind] - layout_detector_array[col_ind])**2, axis=1))
    
    # Create detailed matching information
    source_matches = []
    for i, (snirf_idx, layout_idx) in enumerate(zip(*source_assignments)):
        snirf_source = snirf_data["sources"][snirf_idx]
        layout_source = layout_data["sources"][layout_idx]
        transformed_pos = best_source_transform["func"](np.array([snirf_source["pos"]]))[0]
        
        source_matches.append({
            "snirf_index": snirf_source["index"],
            "layout_index": layout_source["layout_index"],
            "group": layout_source["group"],
            "pos_in_group": layout_source["pos_in_group"],
            "snirf_pos": snirf_source["pos"],
            "transformed_pos": transformed_pos.tolist(),
            "layout_pos": layout_source["pos"],
            "error": source_errors[i]
        })
    
    detector_matches = []
    for i, (snirf_idx, layout_idx) in enumerate(zip(*detector_assignments)):
        snirf_detector = snirf_data["detectors"][snirf_idx]
        layout_detector = layout_data["detectors"][layout_idx]
        transformed_pos = best_detector_transform["func"](np.array([snirf_detector["pos"]]))[0]
        
        detector_matches.append({
            "snirf_index": snirf_detector["index"],
            "layout_index": layout_detector["layout_index"],
            "group": layout_detector["group"],
            "pos_in_group": layout_detector["pos_in_group"],
            "snirf_pos": snirf_detector["pos"],
            "transformed_pos": transformed_pos.tolist(),
            "layout_pos": layout_detector["pos"],
            "error": detector_errors[i]
        })
    
    # Sort matches by layout_index for cleaner output
    source_matches.sort(key=lambda x: x["layout_index"])
    detector_matches.sort(key=lambda x: x["layout_index"])
    
    return {
        "source_transform": best_source_transform["name"],
        "detector_transform": best_detector_transform["name"],
        "source_avg_error": np.mean(source_errors),
        "detector_avg_error": np.mean(detector_errors),
        "source_max_error": np.max(source_errors),
        "detector_max_error": np.max(detector_errors),
        "source_matches": source_matches,
        "detector_matches": detector_matches
    }

def generate_match_report(match_results, snirf_file):
    """Generate a readable report of the matching results."""
    report = f"# Position Matching: {snirf_file} to layout.json\n\n"
    
    # Transformation information
    report += "## Coordinate Transformations\n\n"
    report += f"- **Source Transform**: {match_results['source_transform']}\n"
    report += f"- **Detector Transform**: {match_results['detector_transform']}\n\n"
    
    # Matching quality
    report += "## Matching Quality\n\n"
    report += "### Source Matching\n\n"
    report += f"- Average error: {match_results['source_avg_error']:.4f} units\n"
    report += f"- Maximum error: {match_results['source_max_error']:.4f} units\n\n"
    
    report += "### Detector Matching\n\n"
    report += f"- Average error: {match_results['detector_avg_error']:.4f} units\n"
    report += f"- Maximum error: {match_results['detector_max_error']:.4f} units\n\n"
    
    # Summary of source groups covered
    source_groups = set([m["group"] for m in match_results["source_matches"]])
    report += "## Source Groups Coverage\n\n"
    report += f"The SNIRF file covers {len(source_groups)} source groups from layout.json:\n\n"
    report += ", ".join([str(g) for g in sorted(source_groups)]) + "\n\n"
    
    # Summary of detector groups covered
    detector_groups = set([m["group"] for m in match_results["detector_matches"]])
    report += "## Detector Groups Coverage\n\n"
    report += f"The SNIRF file covers {len(detector_groups)} detector groups from layout.json:\n\n"
    report += ", ".join([str(g) for g in sorted(detector_groups)]) + "\n\n"
    
    # Detailed source matching
    report += "## Source Matching Details\n\n"
    report += "| SNIRF Index | Layout Index | Group | Pos in Group | Error |\n"
    report += "|-------------|--------------|-------|--------------|-------|\n"
    
    for match in match_results["source_matches"]:
        report += f"| {match['snirf_index']} | {match['layout_index']} | {match['group']} | {match['pos_in_group']} | {match['error']:.4f} |\n"
    
    report += "\n"
    
    # Detailed detector matching
    report += "## Detector Matching Details\n\n"
    report += "| SNIRF Index | Layout Index | Group | Pos in Group | Error |\n"
    report += "|-------------|--------------|-------|--------------|-------|\n"
    
    for match in match_results["detector_matches"][:20]:  # Show only first 20 for brevity
        report += f"| {match['snirf_index']} | {match['layout_index']} | {match['group']} | {match['pos_in_group']} | {match['error']:.4f} |\n"
    
    if len(match_results["detector_matches"]) > 20:
        report += f"| ... | ... | ... | ... | ... |\n"
        report += f"(Showing 20 of {len(match_results['detector_matches'])} detector matches)\n\n"
    
    # Worst matches (largest errors)
    report += "## Worst Source Matches\n\n"
    worst_sources = sorted(match_results["source_matches"], key=lambda x: x["error"], reverse=True)[:10]
    
    report += "| SNIRF Index | Layout Index | Error | SNIRF Position | Layout Position |\n"
    report += "|-------------|--------------|-------|----------------|----------------|\n"
    
    for match in worst_sources:
        snirf_pos = f"({match['snirf_pos'][0]:.2f}, {match['snirf_pos'][1]:.2f}, {match['snirf_pos'][2]:.2f})"
        layout_pos = f"({match['layout_pos'][0]:.2f}, {match['layout_pos'][1]:.2f}, {match['layout_pos'][2]:.2f})"
        report += f"| {match['snirf_index']} | {match['layout_index']} | {match['error']:.4f} | {snirf_pos} | {layout_pos} |\n"
    
    report += "\n"
    
    # Group matching visualization
    report += "## Group Coverage Visualization\n\n"
    report += "### Sources\n\n"
    for i in range(1, max([s["group"] for s in layout_data["sources"]]) + 1):
        matched = i in source_groups
        marker = "✅" if matched else "❌"
        report += f"{marker} Group {i}\n"
    
    report += "\n### Detectors\n\n"
    for i in range(1, max([d["group"] for d in layout_data["detectors"]]) + 1):
        matched = i in detector_groups
        marker = "✅" if matched else "❌"
        report += f"{marker} Group {i}\n"
    
    return report

def main():
    """Main function to match positions between SNIRF files and layout.json."""
    parser = argparse.ArgumentParser(description='Match SNIRF positions to layout.json')
    parser.add_argument('--snirf', default=None, help='Path to specific SNIRF file to analyze')
    parser.add_argument('--layout', default="/Users/clem/Desktop/code/data-pipeline/ref/layout.json", 
                        help='Path to layout.json file')
    args = parser.parse_args()
    
    # Load layout.json
    global layout_data  # Make available to report function
    layout_data = load_layout_json(args.layout)
    
    # Find SNIRF files to process
    if args.snirf:
        snirf_files = [Path(args.snirf)]
    else:
        snirf_dir = "/Users/clem/Desktop/code/data-pipeline/ref/snirfs"
        snirf_files = list(Path(snirf_dir).glob("*.snirf"))
    
    print(f"Found {len(snirf_files)} SNIRF files. Matching with layout.json...")
    
    for snirf_file in snirf_files:
        print(f"Processing {snirf_file.name}...")
        snirf_data = extract_snirf_positions(snirf_file)
        
        if snirf_data.get("error"):
            print(f"Error processing {snirf_file.name}: {snirf_data['error']}")
            continue
        
        # Match positions
        match_results = match_positions(snirf_data, layout_data)
        
        # Generate and save report
        report = generate_match_report(match_results, snirf_file.name)
        output_dir = snirf_file.parent
        report_file = output_dir / f"match_{snirf_file.stem}.md"
        
        with open(report_file, "w") as f:
            f.write(report)
        
        print(f"Match report written to: {report_file}")
        
        # Save detailed match data in JSON
        json_file = output_dir / f"match_{snirf_file.stem}.json"
        with open(json_file, "w") as f:
            json.dump(match_results, f, indent=2)
        
        print(f"Match data saved to: {json_file}")
    
    # Generate combined report
    combine_reports(Path(snirf_files[0]).parent)

def combine_reports(output_dir):
    """Combine all matching reports into a single summary file."""
    report_files = list(output_dir.glob("match_*.md"))
    
    combined_report = "# SNIRF to layout.json Matching Summary\n\n"
    combined_report += f"Position matching for {len(report_files)} SNIRF files\n\n"
    combined_report += "## Table of Contents\n\n"
    
    for report_file in report_files:
        file_name = report_file.name.replace("match_", "").replace(".md", "")
        combined_report += f"- [{file_name}](#{file_name.lower().replace('.', '').replace('_', '-')})\n"
    
    combined_report += "\n"
    
    for report_file in sorted(report_files):
        with open(report_file, "r") as f:
            content = f.read()
        
        combined_report += content + "\n---\n\n"
    
    # Write combined report
    combined_report_path = output_dir / "position_matching_summary.md"
    with open(combined_report_path, "w") as f:
        f.write(combined_report)
    
    print(f"Combined summary written to: {combined_report_path}")

if __name__ == "__main__":
    main()