# CLAUDE.md
This file involves the LANG transform

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

really look at t2A, t2B, t2C, t4A, t6A, the base transform, cli.py, query_helpers, lang_processing - all files in lang_processing and neural_processing
i want to change the lang transform to do the following.
for each session, and each tokenizer, i want it to return what it currently returns but as a zarr store instead of an h5. then it should save to s3 zarr with the base transforms infrastructure, but ensuring that root metadata is consolidated. it should create one store per session and per tokenizer. so there should be separate zarr stores for session X with tokenizer Y and session X with tokenizer Z, and session A with tokenizer Y and A with tokenizer Z

so all information within the h5 should be retained, just now it should be in zarr instead, and the tokenizer becomes part of the metadata, or some kind of extra thing. 

really read through how t2C does this, and also base transform. read through all of t2B now to see how it is currently doing it, and what must be changed (including within lang_processing, if anything there)

then write a proposal for every single thing that needs to be changed, what code to change to what and where and why, for each thing, super precise. we never use xarray, only zarr 3. think about the simplest, easiest for future development way to do this, how to make it as clear and simple as possible. really look hard at t4A, how those queries work, because we'll want to easily be able to query the lang zarr just as we query the elements ones, for example.

write the proposal to simple-lang.txt

think deeply, really comb through the code, ultrathink



