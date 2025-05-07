#!/usr/bin/env python
import os
import sys
import logging

# Add the project root to the path to import the necessary modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import the validate transform
from transforms.t2D_validate_v0 import ValidateTransform
from base_transform import BaseTransform

# Setup logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_keep_local')

def test_keep_local_flag():
    """Test if the keep_local flag is correctly passed and used in the transform."""
    
    # Create a simple argparse Namespace with the necessary attributes
    class Args:
        source_prefix = 'curated-h5/'
        dest_prefix = 'processed/validate/'
        s3_bucket = 'conduit-data-dev'
        verbose = True
        log_file = None
        dry_run = True
        keep_local = True
        max_depth = 10
        test = True
        
    args = Args()
    
    logger.info("Creating transform instance with keep_local=True")
    transform = ValidateTransform.from_args(args)
    
    # Log the values of the flags
    logger.info(f"transform.keep_local = {transform.keep_local}")
    logger.info(f"transform.dry_run = {transform.dry_run}")

    # Try overriding directly
    logger.info("Setting keep_local attribute directly")
    transform.keep_local = True
    logger.info(f"After direct set: transform.keep_local = {transform.keep_local}")

if __name__ == "__main__":
    test_keep_local_flag()