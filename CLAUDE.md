# CLAUDE.md
THIS IS A NEW FILE

This fi	le provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run Commands
- Run CLI: `python -m data_pipeline.cli [transform] [arguments]`
- List transforms: `python -m data_pipeline.cli --list-transforms`
- Run specific test: `python ref/test_eeg_preprocessing.py [session_id]`
- Run transform with verbose logging: `python -m data_pipeline.cli [transform] --verbose`
- Running with dry-run mode: `python -m data_pipeline.cli [transform] --dry-run`
- Process only new sessions: `python -m data_pipeline.cli [transform] --new-only`

## Repository Structure
- `base_transform.py`: Core framework for data transformation pipeline stages
- `transforms/`: Transform implementations (e.g. curate, window, lang)
- `utils/`: Utility modules for AWS, logging, metrics, and session handling
- `ref/`: Reference scripts and analysis tools for data inspection

## Transform Pipeline Architecture
- `BaseTransform`: Handles S3 operations, session management, and pipeline orchestration
- Child classes (like `CurateTransform`, `LangTransform`) implement business logic
- Each transform defines SOURCE_PREFIX, DEST_PREFIX, and overrides process_session()
- New transforms must implement from_args() and add_subclass_arguments()

## Code Style
- Typing: Use type hints for function parameters and return values
- Imports: Group as standard library, third-party, local modules
- Error handling: Use try/except with specific exceptions and detailed logging
- Logging: Use utils.logging.setup_logging for consistent logging
- Naming: snake_case for variables/functions, CamelCase for classes
- PEP8 compatible: 4-space indentation, max line length ~80-100
- Docstrings: Google style docstrings format

## AWS Integration
- Session data stored in S3 bucket 'conduit-data-dev'
- Transform metadata stored in DynamoDB 'conduit-pipeline-metadata'
- Use utils.aws for S3 and DynamoDB client initialization
- Script versioning handled by transform framework automatically

we use zarr 3, we never use zarr 2, we never use xarray

## Creating New Transforms
1. Create a new file in `transforms/` directory (e.g., `t3C_new_transform_v0.py`)
2. Extend BaseTransform and set SOURCE_PREFIX and DEST_PREFIX
3. Implement process_session() method to handle individual session processing
4. Implement from_args() and add_subclass_arguments() class methods
5. Add entry point for command-line usage with run_from_command_line()


Your task

first comb through every file in the repo - all transforms, cli, base transform, every util, etc. look at the scripts in ref


really look at t4A and query helpers, base transform, etc

I want to extract any torch stuff into a util dataloader in utils, that takes in a labeled zarr store of multiple sessions, and just returns the labeled dataset. so i can train on all the sessions in the dataset. i want to first train a classifier, so keep that in mind, although the dataloader should be very flexible.

then, i want to create a transform t6B_test_class.py
that is a simple classifier trained on eyes open/closed data across all sessions. so it should load the data from the dataloader util, train a very simple classifier on those windows. look at the shape of the windows in t2A so you know the shape. then remember it should just load the data, and train on it, return the train loss and accuracy on the train/validation split
first, create a proposal called simple-loader.txt that just proposes exactly what to put in the dataloader. currently, much of this is in t4A, but we want to take it out of t4A and out of query helpers, so separate those concerns.

write that proposal for now, super simple, ultrathink

