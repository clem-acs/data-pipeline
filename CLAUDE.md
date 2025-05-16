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

i want to change t2B and t2C to output zarr files instead of hdf5. let's start with t2C

what is the simplest possible way of changing them from uploading/saving an h5, to using the infrastructure in the base transform to save the same exact format but in zarr?

currently, i did this for t2C, but there are some issues i want to fix. for one, elements should also have task_type attributes. they currently do not, which is a problem. also, see the below analysis (based on output from zarbug2.txt or zarrbug2.txt, which you have full access to)
Here’s what the file actually contains, what’s missing vs your design spec, and why the structure probably isn’t what you intended.

---

## 1. What you built (observed on-disk)

| logical table | row-count (1-D length) | where the columns live                        | dtype pattern                                                      |
| ------------- | ---------------------- | --------------------------------------------- | ------------------------------------------------------------------ |
| **elements**  | 448 rows               | flat arrays whose names start with `element_` | mix of `float64`, `int32/int8`, **object** (variable-length UTF-8) |
| **segments**  | 155 rows               | flat arrays whose names start with `segment_` | same                                                               |
| **tasks**     | 32 rows                | flat arrays whose names start with `task_`    | same                                                               |

Everything sits in the *root* group of the v3 store; there are **no sub-groups called `elements/`, `segments/`, `tasks/`**.
Because of that, xarray (and most other tools) see **one huge table with 44 individual 1-D arrays** rather than three distinct datasets.

```text
/
 ├─ element_audio_mode        (448,)
 ├─ element_duration          (448,)
 ├─ …                         …
 ├─ segment_duration          (155,)
 ├─ …
 └─ task_duration             (32,)
```

### Cross-links

* Relationships such as “which segments belong to an element” are duplicated two ways:

  * an *attribute* on the root (`element_…_segments=[…]`) – one big dict-of-lists;
  * a *column* `segment_containing_element_id` (155-long).

That redundancy won’t hurt right now, but the attribute version will blow up when you have 10⁵ + elements and will be painful to edit in-place.

### Codecs / dtype choices

* `dtype=object` ➜ Zarr’s **experimental** `vlen-utf8` codec, hence the warnings.
* All numeric fields defaulted to `float64 / int32`, which is fine, but string columns would be safer / faster if you:

  ```python
  vlen = zarr.string_dtype(encoding="utf-8")   # no UserWarning
  grp.array("audio_mode", data=…, dtype=vlen, chunks=(…) )
  ```

---

## 2. What looks off compared to your spec

| expected                                                                            | status in the file                                                         | notes                                    |
| ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ---------------------------------------- |
| **Three separate datasets / groups**                                                | *missing* — everything is flat                                             | sub-groups never created                 |
| Element columns: `audio_mode`, `type`, `input_modality`, `id`, …                    | **present**                                                                | count correct (448)                      |
| Task columns: `task_type`, `title`, `sequence_idx`, …                               | `task_task_type` present; **`task_title`, `task_sequence_idx` are absent** | probably dropped in the transform        |
| Segment columns: `segment_type`, `task_id`                                          | `segment_segment_type` present; **no task\_id**                            | `segment_type_index` duplicates the type |
| One clear foreign-key column per table (`element.task_id`, `segment.element_id`, …) | `element_task_id` & `segment_containing_element_id` **present**            | OK                                       |
| No duplicated linkage stored in attrs                                               | duplicated, see cross-links above                                          | cleanse in next version                  |

---

## 3. Why it happened

The transform that writes the store evidently did something like:

```python
root = zarr.open(store, mode="w")
root.array("element_audio_mode", …)
root.array("task_audio_mode", …)
```

Instead of:

```python
zg = root.require_group("element")
zg.array("audio_mode", …)
```

Because *Zarr v3* does not forbid flat naming, it happily created 44 top-level arrays; xarray then chokes because it expects a *dataset* with a consistent set of dimensions, not a bag of unrelated vectors.

---

## 4. Fixing / future-proofing

1. **Write hierarchical groups**

   ```python
   root = zarr.open(store, mode="w", version=3)

   g_elem = root.require_group("elements")
   g_task = root.require_group("tasks")
   g_seg  = root.require_group("segments")

   g_elem.array("id", ids, dtype=vlen, chunks=(n_elems,))
   …
   ```

   *Pro:* keeps namespaces clean, lets xarray/dask load each table separately, avoids accidental name clashes.
   *Con:* one more path component when you open arrays, negligible overhead.

2. **Normalise strings**

   Use `zarr.string_dtype` (or fixed-length `dtype="S32"`) to banish the `vlen-utf8` warnings and improve cross-lang portability.

3. **Drop redundant attributes**

   Keep the relational columns (`segment_containing_element_id`, `element_task_id`) and strip the massive `element_*_segments` lists from the root attrs; they aren’t random-access friendly.

4. **Consider Arrow-style tables**

   If you often load whole tables into pandas, a single *struct* array per group (`dtype=[('id','S32'), ('start','f8'), …]`) is another option, but it complicates partial-column reads. The current *columnar* layout is fine for ML pipelines that pull only a couple of fields at a time.

5. **xarray compatibility**

   xarray’s v3 backend is still experimental (2025-05-16). Until it stabilises you can:

   * down-convert to **Zarr v2** (`zarr.copy_store(src, dst, if_exists="replace", version=2)`), or
   * stick with raw zarr + pandas and defer xarray.

---

## 5. Concrete next step

*Rewrite the event transform so it saves to*:

```
/
 ├─ elements/
 │    ├─ id              (N_elem,)
 │    ├─ start_time      …
 │    └─ …
 ├─ tasks/
 │    └─ …
 └─ segments/
      └─ …
```


can you write out here the simplest possible changes that would fix these issues? report back, ultrathink
