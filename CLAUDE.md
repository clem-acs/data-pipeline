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

look especially at the t2 transforms, the folders some of them use, and the outputs they produce (zarr, zarrays, hdf5s)

I want to have a t4A which has various queries in it that we can run in the cli as arguments. for example, I want it to query like for all sessions, give me all element data, with session id as a new dimension
so all this query would do is go through all the existing session by session zarr stores, and return one larger element zar store with dims session_id,element_id, and identical data to what it has currently within the data (so all of the stuff like start/end times, duration, task type, task id, audio mode, absolutely everything)

this is one query; query results should each save with the results in a new zarr store

another query might be like across all sessions, return all elements with task type 'eye' just like session_id,element_id as ims, then just element id, start time, end time as the only data values

the transform t4A should first check if the query result already exists and has all sessions included. if it does, it should just tell the user it's already there (and where). if not, but the result does exist for some sessions, it should simply append the new sessions results. if it's not there, it should start from scratch. eventually, there'll be lots and lots of complex queries in here.
for example, i might want to get all the windows of neural data that occurred across all sessions during elements where the task type is 'eye', and have them return a torch dataset to me in python where it's just the windows (so sliced out of each session) and a label 'closed' if the element_id includes the word 'closed' and an label 'open' if the element_id includes open and 'intro' if element id includes the word intro.
so then it'll return to me a torch dataset of that labeled neural data for training a classifier
so that query should be able to run from here, just as a simpler element-only query, which will save to another zarr store in s3

 i'll want to run this transform like:

python cli.py qry --[normal cli args] --query_name 
where query_name might be like --eye for the eye one, or --all-elements or others. i'll add lots

now, it worked. i ran something, and saved in the s3 in processed/queries/eye_neural.zarr
but then the label map i gave was slightly wrong, so i want to rerun it again, with my new label map. it didn't work, because there was already something there in the folder when it tried to write.

i do want it not to rerun queries on a session if they've already been run on that session. but they should overrun - write over the result -- if i put the flag --include-processed or if the label map is different. they should write over the zarr store with the new info. analyze base transform, cli.py t4a, the other transforms to understand why this isn't happening correctly. what is the current flow, why is this not working. don't change anything yet, just report back ultrathink
look in i11.txt, which you have full access to, to understand the error flow
