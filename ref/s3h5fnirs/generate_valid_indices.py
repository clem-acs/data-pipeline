#!/usr/bin/env python3
"""
Script to generate valid fNIRS channel indices for source-detector combinations
where both source and detector modules are present in the input list.

Each module contains 3 sources and 6 detectors, and each source-detector pair
can have 6 channels (2 wavelengths × 3 moments).
"""

import argparse
import numpy as np


def calculate_channel_index(wavelength_idx, moment_idx, source_module, source_id, detector_module, detector_id):
    """
    Calculate the index for a specific channel based on the formula:
    
    index = ((((wavelength_idx * 3 + moment_idx) * 48 + (source_module-1)) * 3 + (source_id-1)) * 48 + (detector_module-1)) * 6 + (detector_id-1)
    
    Parameters:
    - wavelength_idx: 0 (Red) or 1 (IR)
    - moment_idx: 0 (Zeroth), 1 (First), or 2 (Second)
    - source_module: 1-48
    - source_id: 1-3
    - detector_module: 1-48
    - detector_id: 1-6
    
    Returns:
    - index: The calculated channel index
    """
    index = ((((wavelength_idx * 3 + moment_idx) * 48 + (source_module-1)) * 3 + (source_id-1)) * 48 + (detector_module-1)) * 6 + (detector_id-1)
    return index


def generate_valid_indices(present_modules):
    """
    Generate all valid channel indices for source-detector combinations where
    both source and detector modules are present in the input list.
    
    Parameters:
    - present_modules: List of module numbers that are present (1-48)
    
    Returns:
    - indices: List of valid channel indices
    """
    indices = []
    
    # Iterate through all wavelengths (0=Red, 1=IR)
    for wavelength_idx in range(2):
        # Iterate through all moments (0=Zeroth, 1=First, 2=Second)
        for moment_idx in range(3):
            # Iterate through all present source modules
            for source_module in present_modules:
                # Iterate through all source IDs in the module (1-3)
                for source_id in range(1, 4):
                    # Iterate through all present detector modules
                    for detector_module in present_modules:
                        # Iterate through all detector IDs in the module (1-6)
                        for detector_id in range(1, 7):
                            # Calculate the index for this channel
                            index = calculate_channel_index(
                                wavelength_idx, moment_idx,
                                source_module, source_id,
                                detector_module, detector_id
                            )
                            indices.append(index)
    
    return sorted(indices)


def generate_channel_names(present_modules):
    """
    Generate the channel names for all valid source-detector combinations.
    
    Parameters:
    - present_modules: List of module numbers that are present (1-48)
    
    Returns:
    - channel_names: List of channel names
    """
    wavelength_names = ["Red", "IR"]
    moment_names = ["Zeroth", "First", "Second"]
    channel_names = []
    
    for wavelength_idx, wavelength_name in enumerate(wavelength_names):
        for moment_idx, moment_name in enumerate(moment_names):
            for source_module in present_modules:
                for source_id in range(1, 4):
                    for detector_module in present_modules:
                        for detector_id in range(1, 7):
                            # Format: W{wavelength_idx}({wavelength_name})_M{moment_idx}({moment_name})_S{source_module}_{source_id}_D{detector_module}_{detector_id}
                            channel_name = f"W{wavelength_idx}({wavelength_name})_M{moment_idx}({moment_name})_S{source_module}_{source_id}_D{detector_module}_{detector_id}"
                            channel_names.append(channel_name)
    
    return channel_names


def indices_by_category(present_modules):
    """
    Generate organized indices by wavelength and moment categories.
    
    Parameters:
    - present_modules: List of module numbers that are present (1-48)
    
    Returns:
    - categorized_indices: Dictionary with wavelength and moment categories
    """
    categorized_indices = {
        "wavelengths": {
            "Red": [],   # wavelength_idx = 0
            "IR": []     # wavelength_idx = 1
        },
        "moments": {
            "Zeroth": [],  # moment_idx = 0
            "First": [],   # moment_idx = 1
            "Second": []   # moment_idx = 2
        },
        "wavelength_moment_combinations": {
            "Red_Zeroth": [],
            "Red_First": [],
            "Red_Second": [],
            "IR_Zeroth": [],
            "IR_First": [],
            "IR_Second": []
        }
    }
    
    wavelength_names = ["Red", "IR"]
    moment_names = ["Zeroth", "First", "Second"]
    
    for wavelength_idx, wavelength_name in enumerate(wavelength_names):
        for moment_idx, moment_name in enumerate(moment_names):
            for source_module in present_modules:
                for source_id in range(1, 4):
                    for detector_module in present_modules:
                        for detector_id in range(1, 7):
                            index = calculate_channel_index(
                                wavelength_idx, moment_idx,
                                source_module, source_id,
                                detector_module, detector_id
                            )
                            
                            # Add to wavelength category
                            categorized_indices["wavelengths"][wavelength_name].append(index)
                            
                            # Add to moment category
                            categorized_indices["moments"][moment_name].append(index)
                            
                            # Add to combination category
                            combo_key = f"{wavelength_name}_{moment_name}"
                            categorized_indices["wavelength_moment_combinations"][combo_key].append(index)
    
    # Sort all lists of indices
    for category in categorized_indices:
        for key in categorized_indices[category]:
            categorized_indices[category][key].sort()
    
    return categorized_indices


def main():
    parser = argparse.ArgumentParser(description="Generate valid fNIRS channel indices for specified modules")
    parser.add_argument("modules", type=int, nargs="+", 
                        help="List of present module numbers (1-48)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file to save indices (optional)")
    parser.add_argument("--names", "-n", action="store_true",
                        help="Generate channel names instead of indices")
    parser.add_argument("--categorize", "-c", action="store_true",
                        help="Categorize indices by wavelength and moment")
    
    args = parser.parse_args()
    
    # Validate module numbers
    present_modules = []
    for module in args.modules:
        if module < 1 or module > 48:
            print(f"Warning: Module number {module} is out of range (1-48). Skipping.")
        else:
            present_modules.append(module)
    
    if not present_modules:
        print("Error: No valid module numbers provided.")
        return
    
    if args.categorize:
        result = indices_by_category(present_modules)
        print(f"Generated categorized indices for modules: {present_modules}")
        
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Saved categorized indices to {args.output}")
        else:
            # Print a summary
            for category, subcategories in result.items():
                print(f"\n{category.capitalize()}:")
                for subcategory, indices in subcategories.items():
                    print(f"  - {subcategory}: {len(indices)} indices")
                    if len(indices) <= 5:
                        print(f"    {indices}")
                    else:
                        print(f"    {indices[:5]} ... (and {len(indices)-5} more)")
    
    elif args.names:
        channel_names = generate_channel_names(present_modules)
        print(f"Generated {len(channel_names)} channel names for modules: {present_modules}")
        
        if args.output:
            with open(args.output, 'w') as f:
                for name in channel_names:
                    f.write(name + '\n')
            print(f"Saved channel names to {args.output}")
        else:
            # Print the first few channel names
            for i, name in enumerate(channel_names[:10]):
                print(name)
            if len(channel_names) > 10:
                print(f"... and {len(channel_names)-10} more channel names")
    
    else:
        indices = generate_valid_indices(present_modules)
        print(f"Generated {len(indices)} indices for modules: {present_modules}")
        
        if args.output:
            # Save indices to the specified output file
            np.save(args.output, np.array(indices))
            print(f"Saved indices to {args.output}")
        else:
            # Print the first few indices
            if len(indices) <= 20:
                print(indices)
            else:
                print(indices[:20])
                print(f"... and {len(indices)-20} more indices")


if __name__ == "__main__":
    main()