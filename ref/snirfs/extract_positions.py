#!/usr/bin/env python3
"""
Script to extract source and detector positions from SNIRF files
and summarize them in a structured format.
"""
import os
import sys
import glob
import json
import numpy as np
import h5py
from pathlib import Path

def extract_positions(file_path):
    """Extract source and detector positions from a SNIRF file."""
    try:
        file_name = os.path.basename(file_path)
        print(f"Processing: {file_name}")
        
        results = {
            "file_name": file_name,
            "sources": [],
            "detectors": [],
            "has_3d": False,
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
                results["has_3d"] = True
                sources = probe['sourcePos3D'][:]
                for i, pos in enumerate(sources):
                    results["sources"].append({
                        "index": i + 1,  # 1-based indexing
                        "type": "3D",
                        "x": float(pos[0]),
                        "y": float(pos[1]),
                        "z": float(pos[2])
                    })
            elif 'sourcePos' in probe:
                sources = probe['sourcePos'][:]
                for i, pos in enumerate(sources):
                    source_info = {
                        "index": i + 1,  # 1-based indexing
                        "type": "2D" if len(pos) == 2 else "3D"
                    }
                    
                    source_info["x"] = float(pos[0])
                    source_info["y"] = float(pos[1])
                    if len(pos) > 2:
                        source_info["z"] = float(pos[2])
                        results["has_3d"] = True
                    
                    results["sources"].append(source_info)
            
            # Extract detector positions
            if 'detectorPos3D' in probe:
                results["has_3d"] = True
                detectors = probe['detectorPos3D'][:]
                for i, pos in enumerate(detectors):
                    results["detectors"].append({
                        "index": i + 1,  # 1-based indexing
                        "type": "3D",
                        "x": float(pos[0]),
                        "y": float(pos[1]),
                        "z": float(pos[2])
                    })
            elif 'detectorPos' in probe:
                detectors = probe['detectorPos'][:]
                for i, pos in enumerate(detectors):
                    detector_info = {
                        "index": i + 1,  # 1-based indexing
                        "type": "2D" if len(pos) == 2 else "3D"
                    }
                    
                    detector_info["x"] = float(pos[0])
                    detector_info["y"] = float(pos[1])
                    if len(pos) > 2:
                        detector_info["z"] = float(pos[2])
                        results["has_3d"] = True
                    
                    results["detectors"].append(detector_info)
            
            # Extract wavelengths if available
            if 'wavelengths' in probe:
                wavelengths = probe['wavelengths'][:]
                results["wavelengths"] = [float(w) for w in wavelengths]
            
        return results
    
    except Exception as e:
        return {
            "file_name": os.path.basename(file_path),
            "error": str(e),
            "sources": [],
            "detectors": []
        }

def generate_summary(all_results):
    """Generate text and markdown summaries from position data."""
    text_summary = "# SNIRF Source and Detector Positions Summary\n\n"
    
    for result in all_results:
        file_name = result["file_name"]
        text_summary += f"## {file_name}\n\n"
        
        if result.get("error"):
            text_summary += f"**Error**: {result['error']}\n\n"
            continue
        
        # File information
        text_summary += f"- **Dimension**: {'3D' if result['has_3d'] else '2D'}\n"
        text_summary += f"- **Sources**: {len(result['sources'])}\n"
        text_summary += f"- **Detectors**: {len(result['detectors'])}\n"
        
        if "wavelengths" in result:
            text_summary += f"- **Wavelengths**: {result['wavelengths']}\n"
        
        # Source positions
        if result['sources']:
            text_summary += "\n### Source Positions\n\n"
            text_summary += "| Index | Type | X | Y | Z |\n"
            text_summary += "|-------|------|------|------|------|\n"
            
            for source in result['sources']:
                z_val = source.get("z", "N/A")
                text_summary += f"| {source['index']} | {source['type']} | {source['x']:.4f} | {source['y']:.4f} | {z_val if z_val == 'N/A' else f'{z_val:.4f}'} |\n"
        
        # Detector positions
        if result['detectors']:
            text_summary += "\n### Detector Positions\n\n"
            text_summary += "| Index | Type | X | Y | Z |\n"
            text_summary += "|-------|------|------|------|------|\n"
            
            for detector in result['detectors']:
                z_val = detector.get("z", "N/A")
                text_summary += f"| {detector['index']} | {detector['type']} | {detector['x']:.4f} | {detector['y']:.4f} | {z_val if z_val == 'N/A' else f'{z_val:.4f}'} |\n"
        
        text_summary += "\n\n"
    
    return text_summary

def main():
    """Process all SNIRF files and generate position summaries."""
    if len(sys.argv) > 1:
        # Process specified files
        snirf_files = sys.argv[1:]
    else:
        # Find all SNIRF files in current directory
        snirf_files = glob.glob("*.snirf")
        
        if not snirf_files:
            print("No SNIRF files found in the current directory")
            return
    
    print(f"Processing {len(snirf_files)} SNIRF files...")
    
    all_results = []
    for file_path in snirf_files:
        result = extract_positions(file_path)
        all_results.append(result)
    
    # Generate text summary
    text_summary = generate_summary(all_results)
    
    # Write summaries to files
    with open("snirf_positions_summary.md", "w") as f:
        f.write(text_summary)
    
    with open("snirf_positions_data.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nSummaries written to:")
    print("- snirf_positions_summary.md - Readable markdown summary")
    print("- snirf_positions_data.json - JSON data file")

if __name__ == "__main__":
    main()