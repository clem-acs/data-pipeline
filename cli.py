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
    from .transforms.t0A_curate_v0 import CurateTransform
    from .transforms.t2A_window_v0 import WindowTransform
    from .transforms.t2C_event_v0 import EventTransform
    from .transforms.t2D_validate_v0 import ValidateTransform
    from .transforms.t2E_analysis_v0 import AnalysisTransform
    from .transforms.t4A_query_v0 import QueryTransform
    from .utils.session_utils import (
        get_all_sessions, display_sessions_summary, select_sessions_interactive,
        get_processed_sessions
    )
    from .utils.aws import init_s3_client
except ImportError:
    # When running as a script
    from transforms.t0A_curate_v0 import CurateTransform
    from transforms.t2A_window_v0 import WindowTransform
    from transforms.t2C_event_v0 import EventTransform
    from transforms.t2D_validate_v0 import ValidateTransform
    from transforms.t2E_analysis_v0 import AnalysisTransform
    from transforms.t2F_audio_v0 import AudioExtractTransform
    from transforms.t4A_query_v0 import QueryTransform
    from transforms.t4B_query_eye_neural_v0 import EyeNeuralTransform
    from transforms.t4C_query_neuro_lang_v0 import NeuroLangTransform
    from transforms.t6B_classify_eeg_v0 import ClassifyEEGTransform

    from utils.session_utils import (
        get_all_sessions, display_sessions_summary, select_sessions_interactive,
        get_processed_sessions
    )
    from utils.aws import init_s3_client


# Import the language transform if available
LangTransform = None
try:
    # When running as an installed package
    try:
        from .transforms.t2B_lang_v0 import LangTransform
    except ImportError:
        # When running as a script
        from transforms.t2B_lang_v0 import LangTransform
    print("Successfully imported LangTransform")
except ImportError as e:
    print(f"Language transform not available: {e}")
    LangTransform = None

TRANSFORMS = {
    'curate': CurateTransform,
    'window': WindowTransform,
    'validate': ValidateTransform,
    'analyze': AnalysisTransform,
    'audio': AudioExtractTransform,
    'event': EventTransform,
    'qry': QueryTransform,
    'qry-eye': EyeNeuralTransform,
    'qry-lang': NeuroLangTransform,
    'classify': ClassifyEEGTransform,

    # Add more transforms as they are implemented
    # 'preprocess': PreprocessTransform,
    # 'extract': FeatureExtractionTransform,
    # 'train': ModelTrainingTransform,
}

# Add language transform if available
if LangTransform is not None:
    TRANSFORMS['lang'] = LangTransform


def get_session_ids(args):
    """
    Get session IDs to process based on CLI arguments.
    
    Supports filtering, interactive selection, and direct session specification.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        List of session IDs to process (not file paths)
    """
    prefix = args.source_prefix
    bucket = args.s3_bucket
    s3_client = init_s3_client()
    
    # Get transform ID for checking processed sessions
    transform_id = None
    if hasattr(args, 'transform'):
        transform_class = TRANSFORMS.get(args.transform)
        if transform_class:
            # Handle the --test flag (combines dry-run and keep-local)
            if hasattr(args, 'test') and args.test:
                args.dry_run = True
                args.keep_local = True
                
            # Create a temporary instance to get transform ID
            temp_transform = transform_class.from_args(args)
            transform_id = temp_transform.transform_id
    
    # Check if we need to find processed sessions
    processed_sessions = None
    
    # Check which filtering flags are set
    include_processed = hasattr(args, 'include_processed') and args.include_processed
    include_skipped = hasattr(args, 'include_skipped') and args.include_skipped
    skipped_only = hasattr(args, 'skipped_only') and args.skipped_only
    
    # Get processed sessions based on flags
    if transform_id:
        if skipped_only:
            # Only get skipped sessions
            processed_sessions = get_processed_sessions(transform_id, include_skipped=False, skipped_only=True)
        elif not (include_processed or include_skipped):
            # Default: get both success and skipped to filter them out
            processed_sessions = get_processed_sessions(transform_id, include_skipped=True, skipped_only=False)
        # If include_processed is True, we don't need to get the list since we're not filtering
    
    print(f"Looking for sessions in s3://{bucket}/{prefix}")
    
    # Get all sessions
    sessions = get_all_sessions(
        s3_client, 
        prefix=prefix, 
        bucket=bucket,
        name_filter=args.session if hasattr(args, 'session') and args.session else None
    )
    
    # No duration filter skipping anymore
    
    if not sessions:
        print(f"No sessions found in s3://{bucket}/{prefix}")
        return []
    
    # Handle the automatic processing of all sessions if --all is specified
    if hasattr(args, 'all') and args.all:
        print(f"Processing all available sessions due to --all flag")
        
        # Filter by minimum duration
        filtered_sessions = {
            folder: data for folder, data in sessions.items()
            if data.get('duration_minutes', 0) >= args.min_duration
        }
        
        # Check if we're processing already processed sessions
        if processed_sessions and include_processed:
            # Count how many processed sessions will be included
            processed_count = sum(1 for folder, data in filtered_sessions.items() 
                               if data['name'] in processed_sessions)
            
            # If there are processed sessions and we're not in list-only mode
            if processed_count > 0 and not (hasattr(args, 'list_sessions') and args.list_sessions):
                print(f"\nNOTE: You've specified --include-processed, so {processed_count} already processed session(s) will be reprocessed.")
                confirmation = input("Do you want to continue? (Y/n): ").strip().lower()
                if confirmation == 'n':
                    print("Operation cancelled. Exiting.")
                    return []
        
        # Filter out already processed sessions by default unless --include-processed is specified
        if processed_sessions and not include_processed:
            filtered_sessions = {
                folder: data for folder, data in filtered_sessions.items()
                if data['name'] not in processed_sessions
            }
            print(f"Filtering out already processed sessions (default behavior)")
        
        if not filtered_sessions:
            if processed_sessions and not include_processed:
                print(f"No new (unprocessed) sessions found with minimum duration of {args.min_duration} minutes")
                print(f"Use --include-processed to also process sessions that have already been processed")
            else:
                print(f"No sessions found with minimum duration of {args.min_duration} minutes")
            return []
            
        sorted_sessions = sorted(
            filtered_sessions.items(), 
            key=lambda x: x[1]['duration_ms'], 
            reverse=True
        )
        
        # Show summary to user
        display_sessions_summary(filtered_sessions, args.min_duration, processed_sessions, 
                             include_processed, include_skipped, skipped_only)
    
    # Handle the processing of a specific session if --session is specified
    elif hasattr(args, 'session') and args.session:
        # Filter sessions based on name 
        name_filtered_sessions = {
            folder: data for folder, data in sessions.items()
            if args.session in data['name']
        }
        
        # Apply minimum duration filter
        filtered_sessions = {
            folder: data for folder, data in name_filtered_sessions.items()
            if data.get('duration_minutes', 0) >= args.min_duration
        }
        
        # For named sessions, respect the include_processed flag for
        # determining whether to process already processed sessions
        if processed_sessions and not include_processed:
            filtered_sessions_for_processing = {
                folder: data for folder, data in filtered_sessions.items()
                if data['name'] not in processed_sessions
            }
            
            # If we have already processed the named session(s), show them but don't process
            if not filtered_sessions_for_processing and filtered_sessions:
                print(f"Found {len(filtered_sessions)} sessions matching '{args.session}', but all have already been processed.")
                print(f"Use --include-processed to reprocess already processed sessions.")
                display_sessions_summary(filtered_sessions, args.min_duration, processed_sessions, 
                                     include_processed, include_skipped, skipped_only)
                return []
            
            filtered_sessions = filtered_sessions_for_processing
        
        if not filtered_sessions:
            # Check different failure cases
            if any(args.session in data['name'] for folder, data in sessions.items()):
                if any(args.session in data['name'] and data.get('duration_minutes', 0) < args.min_duration for folder, data in sessions.items()):
                    print(f"Sessions found matching '{args.session}', but none meet the minimum duration of {args.min_duration} minutes")
                else:
                    # This shouldn't generally happen since we only filter processed sessions when they exist
                    print(f"Sessions found matching '{args.session}', but all have been filtered out")
            else:
                print(f"No sessions found matching: {args.session}")
            return []
            
        # First sort by session name alphabetically (for consistent selection) - DESCENDING order to get the LAST one first
        name_sorted_sessions = sorted(
            filtered_sessions.items(),
            key=lambda x: x[1]['name'],
            reverse=True  # Descending order to get the last session alphabetically
        )
        
        # If multiple sessions match, use the LAST one alphabetically
        if len(name_sorted_sessions) > 1:
            # Only select the LAST one alphabetically
            selected_session = name_sorted_sessions[0]  # First item in reverse sorted list = last alphabetically
            print(f"Multiple sessions match '{args.session}'. Using the last one alphabetically: {selected_session[1]['name']}")
            sorted_sessions = [selected_session]
        else:
            # Just one match, use it
            sorted_sessions = name_sorted_sessions
        
        # Show summary to user with processed status if available
        print(f"Session selection with minimum duration of {args.min_duration} minutes:")
        display_sessions_summary({folder: data for folder, data in sorted_sessions}, args.min_duration, 
                               processed_sessions, include_processed, include_skipped, skipped_only)
    
    # Otherwise, go to interactive selection
    else:
        # Let the user select interactively
        sorted_sessions = select_sessions_interactive(sessions, args.min_duration, processed_sessions, 
                                                   include_processed, include_skipped, skipped_only)
        if not sorted_sessions:
            print("No sessions selected for processing")
            return []
    
    # Extract session IDs from session data
    session_ids = []
    for folder, data in sorted_sessions:
        # Get the session name directly
        session_ids.append(data['name'])
    
    return session_ids


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
        
        # Add session selection arguments to transforms that process sessions
        if hasattr(transform_class, 'find_items_to_process'):
            transform_parser.add_argument('--list-sessions', action='store_true',
                                        help='List available sessions without processing')
            transform_parser.add_argument('--session', type=str,
                                        help='Process a specific session by name')
            transform_parser.add_argument('--all', action='store_true',
                                        help='Process all matching sessions')
            transform_parser.add_argument('--min-duration', type=int, default=0,
                                         help='Minimum session duration in minutes')
    
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
    
    # If --list-sessions is specified, just list sessions and exit
    if hasattr(args, 'list_sessions') and args.list_sessions:
        # Handle the --test flag (combines dry-run and keep-local)
        if hasattr(args, 'test') and args.test:
            print("TEST MODE: Enabling both --dry-run and --keep-local flags")
            args.dry_run = True
            args.keep_local = True
            
        # Create a minimal transform instance to get settings 
        transform = transform_class.from_args(args)
        
        # Use the transform's bucket and prefix settings
        s3_client = init_s3_client()
        sessions = get_all_sessions(
            s3_client, 
            prefix=args.source_prefix, 
            bucket=args.s3_bucket,
            name_filter=args.session if hasattr(args, 'session') and args.session else None
        )
        
        if not sessions:
            print(f"No sessions found in s3://{args.s3_bucket}/{args.source_prefix}")
            return 0
        
        # Check for processed sessions if needed
        processed_sessions = None
        include_processed = hasattr(args, 'include_processed') and args.include_processed
        include_skipped = hasattr(args, 'include_skipped') and args.include_skipped
        skipped_only = hasattr(args, 'skipped_only') and args.skipped_only
        
        if not include_processed:
            # Get processed sessions based on flags
            if skipped_only:
                processed_sessions = get_processed_sessions(transform.transform_id, include_skipped=False, skipped_only=True)
            elif not (include_processed or include_skipped):
                # Default behavior: get both success and skipped to filter them out
                processed_sessions = get_processed_sessions(transform.transform_id, include_skipped=True, skipped_only=False)
        
        # Display sessions with processed status
        display_sessions_summary(sessions, args.min_duration, processed_sessions, 
                               include_processed, include_skipped, skipped_only)
        return 0
    
    # Handle the --test flag (combines dry-run and keep-local)
    if hasattr(args, 'test') and args.test:
        print("TEST MODE: Enabling both --dry-run and --keep-local flags")
        args.dry_run = True
        args.keep_local = True
    
    # Create a transform instance
    transform = transform_class.from_args(args)
    
    # If this transform supports session processing and we have session selection args
    if (hasattr(transform, 'find_items_to_process') and 
        (hasattr(args, 'session') or hasattr(args, 'all'))):
        # Get session IDs (will be filtered based on --new-only and/or --show-processed)
        data_ids = get_session_ids(args)
        
        if not data_ids:
            print("No session IDs to process")
            return 0
            
        print(f"Processing {len(data_ids)} session IDs: {data_ids if len(data_ids) < 5 else data_ids[:5] + ['...']}")
        
        # Run with specified session files
        try:
            # We're using include_processed flag to control whether to process already processed sessions
            # This is always needed here since we've already filtered the data_ids based on the include_processed flag
            
            # BaseTransform uses session_ids as the parameter name
            transform.run_pipeline(
                session_ids=data_ids,
                batch_size=args.batch_size,
                max_items=args.max_items,
                start_idx=args.start_idx,
                include_processed=True  # Always set to true since we've already filtered based on include_processed
            )
            return 0
        except Exception as e:
            print(f"Error running transform: {e}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    else:
        # Run the transform with default item detection
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