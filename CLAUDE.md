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
first comb through every file in the repo - all transforms, cli, base transform, every util, etc. then, look in the ref folder at the h5shape files - there are several, like h5shape2 or similar. look at the other groups in the Language group. there are also R and S groups (and an LR, but that's always empty, so ignore it).
the r and s groups both have word datasets with timestamps, rather than char by char data. look at the files called h5shape.txt and h5shape1.txt as examples of these. compare this to the char datasets in L and W
    /language/R
      [Datasets: 1 total, showing 1]
      words: shape=(4255), type=[('word', 'O'), ('start_timestamp', '<f8'), ('end_timestamp', '<f8'), ('element_id', 'O'), ('mode', 'O'), ('display_reference', 'O')], 0 attributes
    /language/S
      [Datasets: 1 total, showing 1]
      words: shape=(1888), type=[('word', 'O'), ('start_timestamp', '<f8'), ('end_timestamp', '<f8'), ('element_id', 'O'), ('mode', 'O'), ('display_reference', 'O')], 0 attributes


i want to modify lang to also tokenize these language groups. so, when you run lang, it should tokenize and return whatever of these four groups exist so it must have new functionality to tokenize timestamped words, not timestamped chars

write a detailed, step by step proposal for how to do this, how to add this functionality to lang, keeping everything else entirely unchanged.
write all functions i must add, write all changes necessary to current code, write what code should change to what and where, for each and every change
it should be as simple as it can possibly be. it should use methods from hugging face, or transformer, or whatever. never reimplement functionality we can just use. as simple, clean, general, minimal as possible. ensure you correctly track all of what is already in lang.py, never reimplement anything. it's currently working perfectly for L and W, we just need to do the word ones now. write this proposal to simple-words.txt write the full step by step proposal there. don't modify anything else. think hard, ultrathink





