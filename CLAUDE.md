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
and the W and L groups have similar but with chars - have a look in the h5shape and h5shape1 files to see the dataset shapes.

in the W group, there is the typing output of data subjects. i tokenized this already in the lang transform. however, now i want to correct the spelling of it and tokenize the corrected version, so i'd end up with aligned incorrect and correct tokens - i'd have a new dataset of tokens in the output file for lang that is W-correct where W-correct has the correct token, and the timestamps of when the incorrect tokens that were associated with it were written, so it would still have token,seq-number or something,  token-id, start time, end time, just as the incorrect ones. that way i'll have both incorrected and corrected timestamped tokens, and i can line them up. note that the lineup may not be 1:1 - often someone will write 'catttttttttttttttt' because they held down the t, and that is just cat, so many tokens may go to one, or i suppose one to many.

i want to add this functionality to the lang transform. in particular, i want to have a few new functions in the lang_processing folder in a file called correction.py, where correction.py has functionality to take in text, correct it, and then tokenize that text from a list of chars into tokens (by using a standard huggingface function for this, or my tokenizer from the other file there, whatever. obviously tokenizer name must be configurable,  so i can use gpt2 or llama or whatever). finally it must use something, maybe difflab, to align the new tokens with the old incorrect ones, add timestamps, and then they should be saved in all output files of the lang transform as another token dataset: token, sequence num, token id, old tokens associated as a range of seq num, start time, end time

i also want it to return the fraction of tokens that remained identical between the initial and the corrected as a measure of typing correctness.


i was thinking i'd use hugging face, symspell, and difflib, but maybe you have other ideas?

anyway, write me a step by step proposal for doing this as simply as possible, super simple, general code. i want it to be as simple as possible. i do not want to reimplement anything that already exists - i'd just like to use existing stuff.


write that simple proposal to simple-correction.txt
think hard, ultrathink. really read and understand first
