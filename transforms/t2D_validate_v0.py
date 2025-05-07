"""
Validation transform for H5 files in the pipeline.

This transform:
1. Reads curated H5 files from S3 (curated-h5/)
2. Analyzes the file structure to identify all groups and datasets
3. Saves validation report as JSON to processed/validate/ prefix in S3
4. Records metadata in DynamoDB

This is implemented using the BaseTransform architecture.
"""

import os
import sys
import json
import h5py
import numpy as np
from typing import Dict, Any, List, Optional, Set

# Import base transform
from base_transform import BaseTransform, Session


class ValidateTransform(BaseTransform):
    """
    Validation transform for H5 files.
    
    This transform analyzes H5 file structure and produces validation reports:
    1. Identifies all groups and datasets in the file
    2. Records metadata about the file structure
    3. Outputs a JSON validation report
    """
    
    # Define required class attributes for source and destination
    SOURCE_PREFIX = 'curated-h5/'
    DEST_PREFIX = 'processed/validate/'
    
    def __init__(self, max_depth: int = 10, **kwargs):
        """
        Initialize the validation transform.
        
        Args:
            max_depth: Maximum depth to traverse in the H5 file structure
            **kwargs: Additional arguments for BaseTransform
        """
        # Set default transform info if not provided
        transform_id = kwargs.pop('transform_id', 'validate_v0')
        script_id = kwargs.pop('script_id', '2D')
        script_name = kwargs.pop('script_name', 'validate_h5')
        script_version = kwargs.pop('script_version', 'v0')
        
        # Explicitly extract keep_local flag to debug it
        keep_local = kwargs.get('keep_local', False)
        
        # Call parent constructor
        super().__init__(
            transform_id=transform_id,
            script_id=script_id,
            script_name=script_name,
            script_version=script_version,
            **kwargs
        )
        
        # Log the keep_local value
        self.logger.info(f"keep_local value in __init__: {self.keep_local}")
        
        # Check if it matches what was passed
        if self.keep_local != keep_local:
            self.logger.warning(f"keep_local mismatch! Passed: {keep_local}, self: {self.keep_local}")
        
        # Set validation-specific attributes
        self.max_depth = max_depth
        
        self.logger.info(f"Validation transform initialized with:")
        self.logger.info(f"  Max depth: {self.max_depth}")
        if self.keep_local:
            self.logger.info(f"  KEEP LOCAL: Will keep temporary files for inspection")
    
    def process_session(self, session: Session) -> Dict:
        """Process a single session.
        
        This implementation:
        1. Finds the curated H5 file for the session
        2. Analyzes its structure
        3. Creates a validation report
        
        Args:
            session: Session object
            
        Returns:
            Dict with processing results
        """
        session_id = session.session_id
        
        # Add extensive logging about keep_local value 
        self.logger.info(f"Keep local value at start of process_session: {self.keep_local}")
        self.logger.info(f"Dry run value: {self.dry_run}")
        
        # Log attributes of self to debug potential issues
        self.logger.info(f"All attributes of self that might affect keep_local:")
        for attr in ['keep_local', 'dry_run']:
            if hasattr(self, attr):
                self.logger.info(f"  {attr}: {getattr(self, attr)}")
                
        self.logger.info(f"Processing session: {session_id}")
        
        # In curated-h5/, files are always directly in the source prefix
        curated_h5_key = f"{self.source_prefix}{session_id}.h5"
        
        # Check if the file exists
        try:
            self.s3.head_object(Bucket=self.s3_bucket, Key=curated_h5_key)
            self.logger.info(f"Found H5 file: {curated_h5_key}")
        except Exception as e:
            self.logger.error(f"No H5 file found for session {session_id}: {e}")
            
            # Log keep_local state again before returning
            self.logger.info(f"keep_local state before error return: {self.keep_local}")
            
            return {
                "status": "failed",
                "error_details": f"No H5 file found for session {session_id}",
                "metadata": {"session_id": session_id},
                "files_to_copy": [],
                "files_to_upload": []
            }
        
        # Download the H5 file
        local_h5_path = session.download_file(curated_h5_key)
        
        try:
            # Analyze the file structure
            structure = self.analyze_h5_structure(local_h5_path, session_id)
            
            # Create the report
            report = {
                "session_id": session_id,
                "file_key": curated_h5_key,
                "structure": structure
            }
            
            # Save the report as JSON
            report_file_name = f"{session_id}_validation.json"
            local_report_path = session.create_upload_file(report_file_name)
            
            with open(local_report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Define the destination key
            dest_key = f"{self.destination_prefix}{report_file_name}"
            
            # Create metadata
            metadata = {
                "session_id": session_id,
                "top_level_groups": structure.get("top_level_groups", []),
                "total_groups": structure.get("total_groups", 0),
                "total_datasets": structure.get("total_datasets", 0),
                "has_eeg": structure.get("has_eeg", False),
                "has_fnirs": structure.get("has_fnirs", False),
                "modalities": structure.get("modalities", [])
            }
            
            # Log keep_local state for debugging
            self.logger.info(f"keep_local state before success return: {self.keep_local}")
            
            return {
                "status": "success",
                "metadata": metadata,
                "files_to_copy": [],
                "files_to_upload": [(local_report_path, dest_key)]
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing session {session_id}: {e}", exc_info=True)
            
            # Log keep_local state before returning error
            self.logger.info(f"keep_local state before exception return: {self.keep_local}")
            
            return {
                "status": "failed",
                "error_details": str(e),
                "metadata": {"session_id": session_id},
                "files_to_copy": [],
                "files_to_upload": []
            }
        finally:
            # Don't manually cleanup files - let the base class handle it
            self.logger.info(f"keep_local state in finally block: {self.keep_local}")
            pass
    
    def analyze_h5_structure(self, file_path: str, session_id: str) -> Dict:
        """Analyze the structure of an H5 file.
        
        Args:
            file_path: Path to the local H5 file
            session_id: Session ID
            
        Returns:
            Dict with structure information
        """
        self.logger.info(f"Analyzing structure of H5 file for session {session_id}")
        
        structure = {
            "top_level_groups": [],
            "groups": {},
            "total_groups": 0,
            "total_datasets": 0,
            "has_eeg": False,
            "has_fnirs": False,
            "modalities": []
        }
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Get top-level groups
                top_level_groups = list(f.keys())
                structure["top_level_groups"] = top_level_groups
                
                # Track group counts
                group_count = 0
                dataset_count = 0
                
                # Check for modalities by looking at the devices group
                if 'devices' in top_level_groups:
                    devices_group = f['devices']
                    
                    # Check for fnirs devices
                    if 'fnirs' in devices_group:
                        structure["has_fnirs"] = True
                        structure["modalities"].append("fnirs")
                        self.logger.info("fNIRS data detected")
                    
                    # Check for eeg devices
                    if 'eeg' in devices_group:
                        structure["has_eeg"] = True
                        structure["modalities"].append("eeg")
                        self.logger.info("EEG data detected")
                
                # Recursive function to process groups
                def process_group(group, path, depth=0):
                    nonlocal group_count, dataset_count
                    
                    if depth > self.max_depth:
                        self.logger.warning(f"Max depth reached at {path}, stopping traversal")
                        return
                    
                    # Initialize group info
                    group_info = {
                        "type": "group",
                        "attributes": {},
                        "subgroups": [],
                        "datasets": []
                    }
                    
                    # Add group attributes
                    for attr_name, attr_value in group.attrs.items():
                        # Convert numpy values to Python types
                        if isinstance(attr_value, (np.ndarray, np.generic)):
                            attr_value = attr_value.tolist()
                        # Convert bytes to string
                        if isinstance(attr_value, bytes):
                            try:
                                attr_value = attr_value.decode('utf-8')
                            except UnicodeDecodeError:
                                attr_value = str(attr_value)
                                
                        group_info["attributes"][attr_name] = attr_value
                    
                    # Process subgroups
                    for subgroup_name, subgroup in group.items():
                        subgroup_path = f"{path}/{subgroup_name}"
                        
                        if isinstance(subgroup, h5py.Group):
                            group_count += 1
                            group_info["subgroups"].append(subgroup_name)
                            # Recursively process the subgroup
                            process_group(subgroup, subgroup_path, depth + 1)
                        
                        elif isinstance(subgroup, h5py.Dataset):
                            dataset_count += 1
                            dataset_info = {
                                "name": subgroup_name,
                                "shape": subgroup.shape,
                                "dtype": str(subgroup.dtype),
                                "size_bytes": subgroup.nbytes
                            }
                            group_info["datasets"].append(dataset_info)
                    
                    # Store group info
                    structure["groups"][path] = group_info
                
                # Start processing from the root
                for group_name in top_level_groups:
                    group = f[group_name]
                    group_path = f"/{group_name}"
                    
                    if isinstance(group, h5py.Group):
                        group_count += 1
                        process_group(group, group_path)
                
                # Update total counts
                structure["total_groups"] = group_count
                structure["total_datasets"] = dataset_count
                
                self.logger.info(f"Structure analysis complete: {group_count} groups, {dataset_count} datasets")
                self.logger.info(f"Modalities detected: {', '.join(structure['modalities']) or 'none'}")
                
        except Exception as e:
            self.logger.error(f"Error analyzing H5 structure for {session_id}: {e}", exc_info=True)
            raise
        
        return structure
    
    @classmethod
    def add_subclass_arguments(cls, parser):
        """Add validation-specific command-line arguments.
        
        Args:
            parser: ArgumentParser instance
        """
        parser.add_argument('--max-depth', type=int, default=10,
                          help='Maximum depth to analyze in H5 file structure')
    
    @classmethod
    def from_args(cls, args):
        """Create a transform instance from command-line arguments.
        
        Args:
            args: Parsed command-line arguments
            
        Returns:
            Instance of ValidateTransform
        """
        # Extract arguments
        source_prefix = getattr(args, 'source_prefix', cls.SOURCE_PREFIX)
        dest_prefix = getattr(args, 'dest_prefix', cls.DEST_PREFIX)
        
        return cls(
            source_prefix=source_prefix,
            destination_prefix=dest_prefix,
            max_depth=args.max_depth,
            s3_bucket=args.s3_bucket,
            verbose=args.verbose,
            log_file=args.log_file,
            dry_run=args.dry_run,
            keep_local=args.keep_local
        )


# Entry point for running the transform from the command line
if __name__ == "__main__":
    ValidateTransform.run_from_command_line()