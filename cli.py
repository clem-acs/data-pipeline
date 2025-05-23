#!/usr/bin/env python3
"""
Command-line interface for running pipeline transforms.

This script provides a unified interface for running different transforms
in the data processing pipeline.
"""


import argparse
import logging
from utils.logging import setup_logging
import sys
import importlib
from datetime import datetime

try:
    # When running as an installed package
    from .transforms.t0A_curate_v0 import CurateTransform
    from .transforms.t2A_window_v0 import WindowTransform
    from .transforms.t2C_event_v0 import EventTransform
    from .transforms.t2D_validate_v0 import ValidateTransform
    from .transforms.t2E_analysis_v0 import AnalysisTransform
    from .transforms.t4A_query_v0 import QueryTransform
    from .transforms.t6A_encode_v0 import EncodeEEGTransform
    from .transforms.t6B_classify_eeg_v0 import ClassifyEEGTransform
    from .utils.session_utils import (
        get_all_sessions, display_sessions_summary, select_sessions_interactive,
        get_processed_sessions, extract_timestamp_from_name
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
    from transforms.t4D_query_eye_good_v0 import EyeGoodTransform
    from transforms.t6B_classify_eeg_v0 import ClassifyEEGTransform
    # Imports for utils when running as a script
    from utils.session_utils import (
        get_all_sessions, display_sessions_summary, select_sessions_interactive,
        get_processed_sessions, extract_timestamp_from_name
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
    'qry-eye-good': EyeGoodTransform,
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

    # Sort all sessions by recency (newest first) based on timestamp in name.
    # This also sets the default order for interactive selection later on.
    try:
        # Sort items from the sessions dictionary. sessions.items() returns a view.
        sorted_session_items = sorted(
            sessions.items(),  # No need to convert to list explicitly for sorted()
            key=lambda item: extract_timestamp_from_name(item[1]['name']),
            reverse=True  # True for newest first
        )
        # Recreate the sessions dictionary. In Python 3.7+, dict preserves insertion order.
        sessions = dict(sorted_session_items)
    except KeyError:
        # This catch block handles unexpected issues, e.g., if 'name' key is missing from session_data.
        # Based on 'get_all_sessions', 'name' should always be present.
        print("Warning: Could not sort sessions by timestamp due to missing 'name' field in session data. Proceeding with original S3 order.")
        # 'sessions' remains as it was, typically in S3 list order (often alphabetical).
        pass

    # Apply date filtering (--since YYYYMMDD)
    # Note: getattr is used for safety, though args should have these if this code path is reached.
    if getattr(args, 'since', None):
        try:
            since_date_str = args.since
            # Validate date format. strptime throws ValueError if format doesn't match.
            datetime.strptime(since_date_str, '%Y%m%d')

            original_count = len(sessions)
            # Filter sessions: keep if session's date is greater than or equal to since_date_str
            # _extract_timestamp_from_name returns YYYYMMDDHHMMSS or "00000000000000"
            # Slicing [:8] gives the YYYYMMDD part.
            sessions = {
                item_path: data
                for item_path, data in sessions.items()
                if extract_timestamp_from_name(data.get('name', ''))[:8] >= since_date_str
            }
            filtered_out_count = original_count - len(sessions)
            if filtered_out_count > 0:
                print(f"Filtered out {filtered_out_count} sessions with dates before {since_date_str}.")

            if not sessions:
                print(f"No sessions remaining after filtering for dates since {since_date_str}.")
                return []
        except ValueError:
            print(f"Error: Invalid date format for --since argument: '{args.since}'. Please use YYYYMMDD.")
            return []
        except Exception as e:
            print(f"Error during date filtering: {e}") # Catch other potential errors (e.g. unexpected data structure)
            return []

    # Apply session size filtering (--min-session-size, --max-session-size)
    min_size_mb = getattr(args, 'min_session_size', None)
    max_size_mb = getattr(args, 'max_session_size', None)

    if min_size_mb is not None or max_size_mb is not None:
        original_count = len(sessions)

        min_size_bytes = min_size_mb * 1024 * 1024 if min_size_mb is not None else None
        max_size_bytes = max_size_mb * 1024 * 1024 if max_size_mb is not None else None

        size_filter_messages = []
        if min_size_bytes is not None:
             size_filter_messages.append(f"minimum {min_size_mb}MB")
        if max_size_bytes is not None:
             size_filter_messages.append(f"maximum {max_size_mb}MB")
        if size_filter_messages:
            print(f"Applying session size filter: {' and '.join(size_filter_messages)}.")

        filtered_sessions_by_size = {}
        for item_path, data in sessions.items():
            # 'total_size_bytes' is expected from get_session_duration via get_all_sessions
            session_size_bytes = data.get('total_size_bytes')

            if session_size_bytes is None:
                # If session size is unknown, exclude it if any size filter is active.
                # This ensures that only sessions with known sizes matching the criteria are processed.
                # print(f"Warning: Session '{data.get('name', item_path)}' missing 'total_size_bytes'. Excluding from size filtering.")
                continue # Skip this session

            # Check against minimum size if specified
            passes_min_filter = (min_size_bytes is None) or (session_size_bytes >= min_size_bytes)
            # Check against maximum size if specified
            passes_max_filter = (max_size_bytes is None) or (session_size_bytes <= max_size_bytes)

            if passes_min_filter and passes_max_filter:
                filtered_sessions_by_size[item_path] = data

        sessions = filtered_sessions_by_size
        filtered_out_count = original_count - len(sessions)

        if filtered_out_count > 0:
            print(f"Filtered out {filtered_out_count} sessions by size criteria.")

        if not sessions:
            print(f"No sessions remaining after size filtering.")
            return []

    # Final check if sessions list is empty after all filters
    if not sessions:
        print(f"No sessions found matching all specified filter criteria (date, size, duration, etc.).")
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

        # Sort the filtered_sessions by recency (newest first).
        # filtered_sessions contains sessions that have passed preliminary checks (e.g., min_duration).
        sorted_sessions = sorted(
            filtered_sessions.items(),
            key=lambda x: extract_timestamp_from_name(x[1]['name']), # Sort by extracted timestamp
            reverse=True  # True ensures newest sessions are first
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

        # Sort sessions matching the --session argument by recency (newest first).
        # filtered_sessions at this point contains sessions that match the name filter and duration criteria.
        timestamp_sorted_sessions = sorted(
            filtered_sessions.items(),
            key=lambda x: extract_timestamp_from_name(x[1]['name']),
            reverse=True  # True for newest first
        )

        # If multiple sessions match the criteria, select the most recent one.
        if len(timestamp_sorted_sessions) > 1:
            selected_session = timestamp_sorted_sessions[0]  # The first item is the most recent
            print(f"Multiple sessions match '{args.session}'. Using the most recent one: {selected_session[1]['name']}")
            sorted_sessions = [selected_session]  # Process only the most recent matching session
        elif timestamp_sorted_sessions: # Handles one match
            # If only one session matches, or if timestamp_sorted_sessions is already narrowed down.
            sorted_sessions = timestamp_sorted_sessions
        else:
            # This case occurs if filtered_sessions was non-empty but timestamp_sorted_sessions is empty.
            # (e.g. if names exist but timestamps can't be extracted, though unlikely with default).
            # The primary 'if not filtered_sessions:' check earlier should catch most "no sessions" scenarios.
            sorted_sessions = []

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

    # Add GLOBAL arguments for logging *before* subparsers
    # These apply to the CLI application itself.
    # Individual transforms might also have these via BaseTransform.add_arguments,
    # but these ensure logging is set up based on top-level CLI invocation.
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose DEBUG level logging for the entire pipeline run.')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Optional file path to write all logs to (in addition to console).')

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
            transform_parser.add_argument('--since', type=str, default=None,
                                         help='Only include sessions since this date (YYYYMMDD). Filters sessions strictly older than this date.')
            transform_parser.add_argument('--min-session-size', type=float, default=None,
                                         help='Minimum session size in MB. Filters sessions smaller than this value.')
            transform_parser.add_argument('--max-session-size', type=float, default=None,
                                         help='Maximum session size in MB. Filters sessions larger than this value.')

    # Parse arguments
    args = parser.parse_args()

    # --- Setup Logging ---
    # Call setup_logging early, using the global --verbose and --log-file arguments.
    # This configures the root logger (""). All other loggers will inherit from this.
    # Assumes cli.py is run from the 'data-pipeline' directory, making 'utils.logging' importable.
    setup_logging(base_logger_name="",  # Configure the root logger
                  verbose=args.verbose,
                  log_file=args.log_file)

    # Optional: Get a logger for cli.py itself.
    # If cli.py is run as a script, __name__ is "__main__".
    # Giving it a specific name like "cli" makes it identifiable.
    cli_logger = logging.getLogger("cli")
    cli_logger.debug(f"CLI arguments parsed: {args}")
    cli_logger.debug("Root logger configured.")
    # --- End Logging Setup ---

    # If --list-transforms is specified, list available transforms and exit
    if args.list_transforms:
        print("Available transforms:")
        for name, transform_class in TRANSFORMS.items():
            print(f"   {name}")
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
            # Use cli_logger if available, otherwise print to stderr
            # cli_logger should be defined after setup_logging near the start of main()
            logger_to_use = locals().get('cli_logger', None)
            if logger_to_use:
                logger_to_use.error(f"Error running transform: {e}", exc_info=True) # exc_info=True adds traceback to log
            else:
                print(f"Error running transform (cli_logger not found): {e}", file=sys.stderr)

            # Still print to stderr and traceback if verbose, for direct console visibility
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
            # Use cli_logger if available, otherwise print to stderr
            logger_to_use = locals().get('cli_logger', None)
            if logger_to_use:
                logger_to_use.error(f"Error running transform: {e}", exc_info=True) # exc_info=True adds traceback to log
            else:
                print(f"Error running transform (cli_logger not found): {e}", file=sys.stderr)

            # Still print to stderr and traceback if verbose, for direct console visibility
            print(f"Error running transform: {e}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1


if __name__ == "__main__":
    sys.exit(main())
