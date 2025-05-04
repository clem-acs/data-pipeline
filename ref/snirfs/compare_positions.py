#!/usr/bin/env python3
"""
Script to compare source and detector positions between SNIRF files and the ref/layout.json file.
"""
import os
import json
import numpy as np
import h5py
from pathlib import Path

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
    
    # Extract source positions
    source_positions = []
    for group in layout_data.get("source_locations", []):
        for i, pos in enumerate(group):
            source_positions.append({
                "index": len(source_positions) + 1,
                "pos": pos
            })
    
    # Extract detector positions
    detector_positions = []
    for i, pos in enumerate(layout_data.get("detector_locations", [])):
        for j, detector_pos in enumerate(pos):
            detector_positions.append({
                "index": len(detector_positions) + 1,
                "pos": detector_pos
            })
    
    return {
        "sources": source_positions,
        "detectors": detector_positions
    }

def compare_positions(snirf_data, layout_data):
    """
    Compare positions between SNIRF and layout.json.
    Returns differences and statistics.
    """
    results = {
        "source_count_diff": len(snirf_data["sources"]) - len(layout_data["sources"]),
        "detector_count_diff": len(snirf_data["detectors"]) - len(layout_data["detectors"]),
        "source_differences": [],
        "detector_differences": [],
        "missing_sources": [],
        "missing_detectors": [],
        "extra_sources": [],
        "extra_detectors": [],
        "summary": {}
    }
    
    # Create lookup dictionaries for source and detector indices
    snirf_sources = {s["index"]: s["pos"] for s in snirf_data["sources"]}
    layout_sources = {s["index"]: s["pos"] for s in layout_data["sources"]}
    
    snirf_detectors = {d["index"]: d["pos"] for d in snirf_data["detectors"]}
    layout_detectors = {d["index"]: d["pos"] for d in layout_data["detectors"]}
    
    # Calculate source differences
    source_diffs = []
    for idx in snirf_sources:
        if idx in layout_sources:
            snirf_pos = snirf_sources[idx]
            layout_pos = layout_sources[idx]
            diff = [snirf_pos[i] - layout_pos[i] for i in range(3)]
            euclidean_dist = np.sqrt(sum([d**2 for d in diff]))
            
            source_diffs.append({
                "index": idx,
                "snirf_pos": snirf_pos,
                "layout_pos": layout_pos,
                "difference": diff,
                "euclidean_distance": euclidean_dist
            })
        else:
            results["extra_sources"].append({"index": idx, "pos": snirf_sources[idx]})
    
    # Find missing sources
    for idx in layout_sources:
        if idx not in snirf_sources:
            results["missing_sources"].append({"index": idx, "pos": layout_sources[idx]})
    
    # Calculate detector differences
    detector_diffs = []
    for idx in snirf_detectors:
        if idx in layout_detectors:
            snirf_pos = snirf_detectors[idx]
            layout_pos = layout_detectors[idx]
            diff = [snirf_pos[i] - layout_pos[i] for i in range(3)]
            euclidean_dist = np.sqrt(sum([d**2 for d in diff]))
            
            detector_diffs.append({
                "index": idx,
                "snirf_pos": snirf_pos,
                "layout_pos": layout_pos,
                "difference": diff,
                "euclidean_distance": euclidean_dist
            })
        else:
            results["extra_detectors"].append({"index": idx, "pos": snirf_detectors[idx]})
    
    # Find missing detectors
    for idx in layout_detectors:
        if idx not in snirf_detectors:
            results["missing_detectors"].append({"index": idx, "pos": layout_detectors[idx]})
    
    # Sort differences by Euclidean distance
    results["source_differences"] = sorted(source_diffs, key=lambda x: x["euclidean_distance"], reverse=True)
    results["detector_differences"] = sorted(detector_diffs, key=lambda x: x["euclidean_distance"], reverse=True)
    
    # Calculate summary statistics
    if source_diffs:
        source_distances = [d["euclidean_distance"] for d in source_diffs]
        results["summary"]["source_max_distance"] = max(source_distances)
        results["summary"]["source_min_distance"] = min(source_distances)
        results["summary"]["source_avg_distance"] = sum(source_distances) / len(source_distances)
    
    if detector_diffs:
        detector_distances = [d["euclidean_distance"] for d in detector_diffs]
        results["summary"]["detector_max_distance"] = max(detector_distances)
        results["summary"]["detector_min_distance"] = min(detector_distances)
        results["summary"]["detector_avg_distance"] = sum(detector_distances) / len(detector_distances)
    
    return results

def generate_comparison_report(comparison, snirf_file):
    """Generate a readable report of the comparison."""
    report = f"# Position Comparison: {snirf_file} vs layout.json\n\n"
    
    # Source Count Comparison
    report += "## Source Count Comparison\n\n"
    if comparison["source_count_diff"] == 0:
        report += "✅ **Same number of sources** in both files.\n\n"
    else:
        report += f"❌ **Different number of sources**: "
        if comparison["source_count_diff"] > 0:
            report += f"{comparison['source_count_diff']} more sources in SNIRF file.\n\n"
        else:
            report += f"{abs(comparison['source_count_diff'])} more sources in layout.json.\n\n"
    
    # Detector Count Comparison
    report += "## Detector Count Comparison\n\n"
    if comparison["detector_count_diff"] == 0:
        report += "✅ **Same number of detectors** in both files.\n\n"
    else:
        report += f"❌ **Different number of detectors**: "
        if comparison["detector_count_diff"] > 0:
            report += f"{comparison['detector_count_diff']} more detectors in SNIRF file.\n\n"
        else:
            report += f"{abs(comparison['detector_count_diff'])} more detectors in layout.json.\n\n"
    
    # Summary Statistics
    report += "## Position Differences Summary\n\n"
    
    summary = comparison["summary"]
    if "source_avg_distance" in summary:
        report += "### Source Position Differences\n\n"
        report += f"- Average difference: {summary['source_avg_distance']:.4f} units\n"
        report += f"- Maximum difference: {summary['source_max_distance']:.4f} units\n"
        report += f"- Minimum difference: {summary['source_min_distance']:.4f} units\n\n"
    
    if "detector_avg_distance" in summary:
        report += "### Detector Position Differences\n\n"
        report += f"- Average difference: {summary['detector_avg_distance']:.4f} units\n"
        report += f"- Maximum difference: {summary['detector_max_distance']:.4f} units\n"
        report += f"- Minimum difference: {summary['detector_min_distance']:.4f} units\n\n"
    
    # Missing Sources
    if comparison["missing_sources"]:
        report += "## Missing Sources in SNIRF File\n\n"
        report += "Sources that are in layout.json but not in the SNIRF file:\n\n"
        report += "| Index | X | Y | Z |\n"
        report += "|-------|------|------|------|\n"
        for source in comparison["missing_sources"]:
            report += f"| {source['index']} | {source['pos'][0]:.4f} | {source['pos'][1]:.4f} | {source['pos'][2]:.4f} |\n"
        report += "\n"
    
    # Extra Sources
    if comparison["extra_sources"]:
        report += "## Extra Sources in SNIRF File\n\n"
        report += "Sources that are in the SNIRF file but not in layout.json:\n\n"
        report += "| Index | X | Y | Z |\n"
        report += "|-------|------|------|------|\n"
        for source in comparison["extra_sources"]:
            report += f"| {source['index']} | {source['pos'][0]:.4f} | {source['pos'][1]:.4f} | {source['pos'][2]:.4f} |\n"
        report += "\n"
    
    # Missing Detectors
    if comparison["missing_detectors"]:
        report += "## Missing Detectors in SNIRF File\n\n"
        report += "Detectors that are in layout.json but not in the SNIRF file:\n\n"
        report += "| Index | X | Y | Z |\n"
        report += "|-------|------|------|------|\n"
        for detector in comparison["missing_detectors"]:
            report += f"| {detector['index']} | {detector['pos'][0]:.4f} | {detector['pos'][1]:.4f} | {detector['pos'][2]:.4f} |\n"
        report += "\n"
    
    # Extra Detectors
    if comparison["extra_detectors"]:
        report += "## Extra Detectors in SNIRF File\n\n"
        report += "Detectors that are in the SNIRF file but not in layout.json:\n\n"
        report += "| Index | X | Y | Z |\n"
        report += "|-------|------|------|------|\n"
        for detector in comparison["extra_detectors"]:
            report += f"| {detector['index']} | {detector['pos'][0]:.4f} | {detector['pos'][1]:.4f} | {detector['pos'][2]:.4f} |\n"
        report += "\n"
    
    # Largest Position Differences
    if comparison["source_differences"]:
        report += "## Top 10 Largest Source Position Differences\n\n"
        report += "| Index | SNIRF Position | Layout Position | Euclidean Distance |\n"
        report += "|-------|---------------|----------------|-------------------|\n"
        for diff in comparison["source_differences"][:10]:
            snirf_pos = f"({diff['snirf_pos'][0]:.2f}, {diff['snirf_pos'][1]:.2f}, {diff['snirf_pos'][2]:.2f})"
            layout_pos = f"({diff['layout_pos'][0]:.2f}, {diff['layout_pos'][1]:.2f}, {diff['layout_pos'][2]:.2f})"
            report += f"| {diff['index']} | {snirf_pos} | {layout_pos} | {diff['euclidean_distance']:.4f} |\n"
        report += "\n"
    
    if comparison["detector_differences"]:
        report += "## Top 10 Largest Detector Position Differences\n\n"
        report += "| Index | SNIRF Position | Layout Position | Euclidean Distance |\n"
        report += "|-------|---------------|----------------|-------------------|\n"
        for diff in comparison["detector_differences"][:10]:
            snirf_pos = f"({diff['snirf_pos'][0]:.2f}, {diff['snirf_pos'][1]:.2f}, {diff['snirf_pos'][2]:.2f})"
            layout_pos = f"({diff['layout_pos'][0]:.2f}, {diff['layout_pos'][1]:.2f}, {diff['layout_pos'][2]:.2f})"
            report += f"| {diff['index']} | {snirf_pos} | {layout_pos} | {diff['euclidean_distance']:.4f} |\n"
        report += "\n"
    
    return report

def main():
    """
    Compare source and detector positions between SNIRF files and layout.json
    """
    # Load layout.json
    layout_json_path = "/Users/clem/Desktop/code/data-pipeline/ref/layout.json"
    layout_data = load_layout_json(layout_json_path)
    
    # Find all SNIRF files
    snirf_dir = "/Users/clem/Desktop/code/data-pipeline/ref/snirfs"
    snirf_files = list(Path(snirf_dir).glob("*.snirf"))
    
    print(f"Found {len(snirf_files)} SNIRF files. Comparing with layout.json...")
    
    for snirf_file in snirf_files:
        snirf_data = extract_snirf_positions(snirf_file)
        
        if snirf_data.get("error"):
            print(f"Error processing {snirf_file.name}: {snirf_data['error']}")
            continue
        
        comparison = compare_positions(snirf_data, layout_data)
        report = generate_comparison_report(comparison, snirf_file.name)
        
        # Write report to file
        report_file = Path(snirf_dir) / f"comparison_{snirf_file.stem}.md"
        with open(report_file, "w") as f:
            f.write(report)
        
        print(f"Comparison report written to: {report_file}")
    
    # Generate a combined summary
    combine_reports(snirf_dir)

def combine_reports(snirf_dir):
    """Combine all comparison reports into a single summary file."""
    report_files = list(Path(snirf_dir).glob("comparison_*.md"))
    
    combined_report = "# SNIRF vs layout.json Position Comparison Summary\n\n"
    combined_report += f"Comparison of {len(report_files)} SNIRF files with layout.json\n\n"
    combined_report += "## Table of Contents\n\n"
    
    for report_file in report_files:
        file_name = report_file.name.replace("comparison_", "").replace(".md", "")
        combined_report += f"- [{file_name}](#{file_name.lower().replace('.', '').replace('_', '-')})\n"
    
    combined_report += "\n"
    
    for report_file in report_files:
        with open(report_file, "r") as f:
            content = f.read()
        
        combined_report += content + "\n---\n\n"
    
    # Write combined report
    combined_report_path = Path(snirf_dir) / "position_comparison_summary.md"
    with open(combined_report_path, "w") as f:
        f.write(combined_report)
    
    print(f"Combined summary written to: {combined_report_path}")

if __name__ == "__main__":
    main()