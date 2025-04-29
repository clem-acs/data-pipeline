#!/usr/bin/env python3
"""
Command-line interface for running pipeline transforms.

This script provides a unified interface for running different transforms
in the data processing pipeline.
"""

import argparse
import sys
import importlib

try:
    # When running as an installed package
    from .transforms.curate import CurationTransform
except ImportError:
    # When running as a script
    from transforms.curate import CurationTransform


TRANSFORMS = {
    'curate': CurationTransform,
    # Add more transforms as they are implemented
    # 'preprocess': PreprocessTransform,
    # 'extract': FeatureExtractionTransform,
    # 'train': ModelTrainingTransform,
}


def main():
    """Main entry point for the pipeline CLI."""
    # Create main parser
    parser = argparse.ArgumentParser(
        description='Run pipeline transforms for data processing'
    )
    
    # Add main arguments
    parser.add_argument('--list-transforms', action='store_true',
                      help='List available transforms')
    
    # Create subparsers for each transform
    subparsers = parser.add_subparsers(
        dest='transform',
        help='Transform to run'
    )
    
    # Add a subparser for each transform
    for name, transform_class in TRANSFORMS.items():
        transform_parser = subparsers.add_parser(
            name, 
            help=transform_class.__doc__.split('\n')[0]
        )
        transform_class.add_arguments(transform_parser)
        transform_class.add_subclass_arguments(transform_parser)
    
    # Parse arguments
    args = parser.parse_args()
    
    # If --list-transforms is specified, list available transforms and exit
    if args.list_transforms:
        print("Available transforms:")
        for name, transform_class in TRANSFORMS.items():
            print(f"  {name}: {transform_class.__doc__.split('\n')[0]}")
        return 0
    
    # If no transform is specified, print help and exit
    if not args.transform:
        parser.print_help()
        return 1
    
    # Get the specified transform class
    transform_class = TRANSFORMS[args.transform]
    
    # Create a transform instance
    transform = transform_class.from_args(args)
    
    # Run the transform
    try:
        transform.run_pipeline(
            batch_size=args.batch_size,
            max_items=args.max_items,
            start_idx=args.start_idx
        )
        return 0
    except Exception as e:
        print(f"Error running transform: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())