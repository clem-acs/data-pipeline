# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

## Creating New Transforms
1. Create a new file in `transforms/` directory (e.g., `t3C_new_transform_v0.py`)
2. Extend BaseTransform and set SOURCE_PREFIX and DEST_PREFIX
3. Implement process_session() method to handle individual session processing
4. Implement from_args() and add_subclass_arguments() class methods
5. Add entry point for command-line usage with run_from_command_line()


Your task

first comb through every file in the repo - all transforms, cli, base transform, every util, etc. look at the scripts in ref

I want a transform like T2e or something that extracts all the types of event data in various ways

after that, i'll later combine the output of window, lang, and this event transform into a tiledb dataset for very easy large model training

so think about the deserata of the eventual tiledb dataset - super easy, fast querying for training lots of things with just a line or two of code to query. that will be like t4a or something. but we aren't really coding that up yet, we just need to keep that goal in mind and really grok what we need to do for it

read the file tamuz-inspected.txt in full and consider the actual kinds of events at play here. some are saved as two event types, for example (element sent and element replied make up one full element, for example, same with some others), so we'll want to save those differently from one-shot events, like a display event for example. really look at all the data, the shapes, how this is similar to aspects of lang or window, etc.

then look carefully at the t2e transform, which attempts to do this

however, there are issues.
most centrally, there are some null issues in the tasks and in the elements data that cause downstream null pointer issues. however, more pressingly, this is hard to debug because the code was written by my junior, who wrote code in a clunky way that's hard to debug - lots of duplicated code, etc.
write a report to report.txt that explains how to massively simplify/eliminate type issues/get rid of duplicate code
think hard, really read and grok each line, and then write the issues and specific fixes to report.txt ultrathink
