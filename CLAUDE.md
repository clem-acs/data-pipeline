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

now, i want to work on t6B the classifier. look through that carefully, comb through each line. currently, I ran it and got an error, which you can see in e2 (which you have full access to
The run blew up in two places:

| Problem                                                                      | Why it happens                                                                                                                                                                                                                     | Quick patch                                                                                                                        |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **`batch_size` shows as 0** in the hyper-param print-out.                    | After we deleted the duplicate flag, the transform now inherits **BaseTransform’s** global `--batch-size`, whose default is `0` (that flag was meant for “how many sessions to process in one CLI batch”, not ML mini-batch size). | Inside `ClassifyEEGTransform.from_args()` copy the value from `args.batch_size`, but fall back to the model default when it’s `0`. |
| **`storage_options was provided but unused…`** comes from `zarr.open_group`. | We passed a plain relative path (`processed/queries/eye_neural.zarr`). Without the `s3://bucket/` prefix PyZarr treats it as a **directory store**, which rejects `storage_options`.                                               | Build the query path with the S3 scheme so zarr uses s3fs.                                                                         |

Below is a *minimal* diff you can paste over the current file; only the two small blocks are changed.

```diff
@@
     def from_args(cls, args):
         # collect h-params
         hp = dict(DEFAULT_HPARAMS)
-        for k in hp.keys():
-            if hasattr(args, k.replace("-", "_")):
-                hp[k] = getattr(args, k.replace("-", "_"))
+        for k in hp.keys():
+            attr = k.replace("-", "_")
+            if hasattr(args, attr):
+                hp[k] = getattr(args, attr)
+
+        # mini-batch size comes from the pipeline-level --batch-size;
+        # if the user didn’t override it (0) fall back to default.
+        if getattr(args, "batch_size", 0) > 0:
+            hp["batch_size"] = args.batch_size
 
         return cls(
@@
     def process_session(self, session: Session) -> Dict[str, Any]:
         """Train and evaluate the classifier; upload artefacts."""
-        query_zarr  = f"{self.source_prefix}{session.session_id}.zarr"
+        query_zarr  = f"s3://{self.s3_bucket}/{self.source_prefix}{session.session_id}.zarr"
@@
```

#### After the patch

```bash
python cli.py classify ^
   --query eye_neural ^
   --epochs 25 --batch-size 64 2>&1 | tee e1.txt
```

* hyper-params will show `batch_size : 64`
* the data loaders open `s3://conduit-data-dev/processed/queries/eye_neural.zarr`
* training proceeds and finishes by uploading:

```
s3://conduit-data-dev/models/eye_neural_eeg_classifier.pth
s3://conduit-data-dev/models/eye_neural_metrics.json
```

No more “storage\_options” warning or argparse conflicts.


can you evaluate those two proposed fixes and see if you think they will make it work? don't change it yet, just read carefully and report back ultrathink
