import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional, Union
from .window_dataset import WindowDataset # Corrected relative import

# Set up logger
logger = logging.getLogger(__name__)

# Basic console handler setup
if not logger.handlers and not logging.getLogger().handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

def create_windows(
    eeg_data: Optional[np.ndarray] = None,
    fnirs_data: Optional[np.ndarray] = None,
    eeg_timestamps: Optional[np.ndarray] = None,
    fnirs_timestamps: Optional[np.ndarray] = None,
    return_torch_tensors: bool = False,
    metadata: Optional[dict] = None
):
    """
    Creates time-aligned windows from EEG and fNIRS data using a fNIRS-driven rhythm,
    handles missing fNIRS frames by inserting placeholders, truncates end of fNIRS data
    if quality is low, adjusts EEG overlap accordingly, and returns a WindowDataset.
    """

    eeg_frames_per_window = 7
    fnirs_frames_per_window = 1
    fnirs_ideal_period_ms = 210.0
    drift_chunk_size_frames = 20
    min_actual_frame_density_for_truncation = 0.80
    min_frames_for_late_offset_calc = drift_chunk_size_frames

    # --- 1. Validate Inputs and Extract Timestamps ---
    if not (eeg_data is not None and len(eeg_data) > 0 and
            eeg_timestamps is not None and len(eeg_timestamps) > 0):
        logger.error("EEG data or timestamps are missing or empty.")
        return None
    if eeg_data.ndim != 3 or eeg_data.shape[1] == 0 or eeg_data.shape[2] == 0:
        logger.error(f"EEG data has unexpected shape: {eeg_data.shape}.")
        return None

    if not (fnirs_data is not None and len(fnirs_data) > 0 and
            fnirs_timestamps is not None and len(fnirs_timestamps) > 0):
        logger.error("fNIRS data or timestamps are missing or empty.")
        return None
    if fnirs_data.ndim != 3 or fnirs_data.shape[1] == 0 or fnirs_data.shape[2] != 1:
        logger.error(f"fNIRS data has unexpected shape: {fnirs_data.shape}.")
        return None

    try:
        eeg_ts_ms = eeg_timestamps[:, 1].astype(float)
        if eeg_ts_ms.shape[0] != eeg_data.shape[0]:
            logger.error(f"EEG timestamp count ({eeg_ts_ms.shape[0]}) " +
                           f"doesn't match EEG data frame count ({eeg_data.shape[0]}).")
            return None
    except IndexError:
        logger.error("EEG timestamps array does not have a second column (index 1).")
        return None
    except Exception as e:
        logger.error(f"Could not extract or validate EEG timestamps: {e}.")
        return None

    try:
        fnirs_ts_ms_orig = fnirs_timestamps[:, 1].astype(float)
        if fnirs_ts_ms_orig.shape[0] != fnirs_data.shape[0]:
            logger.error(f"fNIRS timestamp count ({fnirs_ts_ms_orig.shape[0]}) " +
                           f"doesn't match fNIRS data frame count ({fnirs_data.shape[0]}).")
            return None
    except IndexError:
        logger.error("fNIRS timestamps array does not have a second column (index 1).")
        return None
    except Exception as e:
        logger.error(f"Could not extract or validate fNIRS timestamps: {e}.")
        return None

    logger.info(f"Successfully extracted {len(eeg_ts_ms)} EEG timestamps and {len(fnirs_ts_ms_orig)} fNIRS timestamps.")

    # --- 1.5. Determine Mutually Overlapping Data Range (on original timestamps) ---
    logger.info("Phase 1.5: Determining mutually overlapping data range...")
    if len(eeg_ts_ms) == 0 or len(fnirs_ts_ms_orig) == 0:
        logger.error("Cannot determine overlap: EEG or fNIRS original timestamps are empty.")
        return None

    min_eeg_ts, max_eeg_ts = eeg_ts_ms[0], eeg_ts_ms[-1]
    min_fnirs_ts_orig, max_fnirs_ts_orig = fnirs_ts_ms_orig[0], fnirs_ts_ms_orig[-1]

    overlap_start_ts = max(min_eeg_ts, min_fnirs_ts_orig)
    overlap_end_ts = min(max_eeg_ts, max_fnirs_ts_orig)

    if overlap_start_ts >= overlap_end_ts:
        logger.error("No temporal overlap between EEG and original fNIRS data.")
        return None
    logger.info(f"Temporal overlap identified: [{overlap_start_ts:.3f} ms, {overlap_end_ts:.3f} ms]")

    first_orig_fnirs_idx_in_overlap = np.searchsorted(fnirs_ts_ms_orig, overlap_start_ts, side='left')
    last_orig_fnirs_idx_in_overlap = np.searchsorted(fnirs_ts_ms_orig, overlap_end_ts, side='right') - 1

    first_eeg_idx_in_overlap = np.searchsorted(eeg_ts_ms, overlap_start_ts, side='left')
    original_last_eeg_idx_in_overlap = np.searchsorted(eeg_ts_ms, overlap_end_ts, side='right') - 1
    last_eeg_idx_in_overlap = original_last_eeg_idx_in_overlap

    if not (first_orig_fnirs_idx_in_overlap <= last_orig_fnirs_idx_in_overlap):
        logger.error("No fNIRS frames fall within the overlapping time range.")
        return None
    num_orig_fnirs_in_overlap = last_orig_fnirs_idx_in_overlap - first_orig_fnirs_idx_in_overlap + 1

    if not (first_eeg_idx_in_overlap <= last_eeg_idx_in_overlap and
            (last_eeg_idx_in_overlap - first_eeg_idx_in_overlap + 1) >= eeg_frames_per_window):
        logger.error("Not enough EEG frames fall within the overlapping time range for a window.")
        return None

    logger.info(f"Original fNIRS frames in overlap: {num_orig_fnirs_in_overlap} " +
                f"(indices {first_orig_fnirs_idx_in_overlap} to {last_orig_fnirs_idx_in_overlap})")
    logger.info(f"Initial EEG frames in overlap: {last_eeg_idx_in_overlap - first_eeg_idx_in_overlap + 1} " +
                f"(indices {first_eeg_idx_in_overlap} to {last_eeg_idx_in_overlap})")

    # --- 1.7. fNIRS Gap Filling and Preprocessing ---
    logger.info("Phase 1.7: Processing fNIRS data for gaps...")
    fnirs_ts_ms_processed_list = []
    fnirs_data_processed_list = []
    fnirs_is_actual_list = []

    if num_orig_fnirs_in_overlap == 0:
        logger.error("Cannot process fNIRS gaps: no original fNIRS frames in overlap.")
        return None

    current_actual_ts = fnirs_ts_ms_orig[first_orig_fnirs_idx_in_overlap]
    fnirs_ts_ms_processed_list.append(current_actual_ts)
    fnirs_data_processed_list.append(fnirs_data[first_orig_fnirs_idx_in_overlap])
    fnirs_is_actual_list.append(True)

    for i in range(first_orig_fnirs_idx_in_overlap, last_orig_fnirs_idx_in_overlap):
        ts_of_current_actual_frame_in_loop = fnirs_ts_ms_orig[i]
        ts_of_next_actual_frame_in_loop = fnirs_ts_ms_orig[i+1]
        time_diff = ts_of_next_actual_frame_in_loop - ts_of_current_actual_frame_in_loop
        num_expected_periods = int(round(time_diff / fnirs_ideal_period_ms))
        if num_expected_periods > 1:
            num_gaps_to_fill = num_expected_periods - 1
            logger.info(f"Gap detected: {time_diff:.2f}ms between original fNIRS idx {i} (ts {ts_of_current_actual_frame_in_loop:.2f}) and " +
                        f"{i+1} (ts {ts_of_next_actual_frame_in_loop:.2f}). Expecting {num_expected_periods} periods, filling {num_gaps_to_fill} gaps.")
            last_added_ts_to_processed_list = fnirs_ts_ms_processed_list[-1]
            for k in range(num_gaps_to_fill):
                placeholder_ts = last_added_ts_to_processed_list + fnirs_ideal_period_ms
                fnirs_ts_ms_processed_list.append(placeholder_ts)
                fnirs_data_processed_list.append(np.zeros(fnirs_data[0].shape, dtype=fnirs_data.dtype))
                fnirs_is_actual_list.append(False)
                logger.debug(f"    Placeholder {k+1}/{num_gaps_to_fill} added with ideal ts: {placeholder_ts:.3f} ms")
                last_added_ts_to_processed_list = placeholder_ts
        fnirs_ts_ms_processed_list.append(ts_of_next_actual_frame_in_loop)
        fnirs_data_processed_list.append(fnirs_data[i+1])
        fnirs_is_actual_list.append(True)

    fnirs_ts_ms = np.array(fnirs_ts_ms_processed_list)
    logger.info(f"About to stack fnirs_data_processed_list. Length: {len(fnirs_data_processed_list)}, Shape of first element: {fnirs_data_processed_list[0].shape if fnirs_data_processed_list else 'N/A'}, Dtype: {fnirs_data_processed_list[0].dtype if fnirs_data_processed_list else 'N/A'}")
    fnirs_data_processed = np.stack(fnirs_data_processed_list)
    fnirs_is_actual_frame = np.array(fnirs_is_actual_list)
    num_total_fnirs_frames_processed = len(fnirs_ts_ms)

    if num_total_fnirs_frames_processed > 0:
        log_ts_list = fnirs_ts_ms_processed_list
        log_actual_list = fnirs_is_actual_list
        logger.debug(f"First 5 processed fNIRS timestamps (pre-truncation): {log_ts_list[:5]}")
        logger.debug(f"Last 5 processed fNIRS timestamps (pre-truncation): {log_ts_list[-5:]}")
        logger.debug(f"First 5 fNIRS is_actual flags (pre-truncation): {log_actual_list[:5]}")
        logger.debug(f"Last 5 fNIRS is_actual flags (pre-truncation): {log_actual_list[-5:]}")

    num_actual_fnirs_frames_processed = np.sum(fnirs_is_actual_frame)
    logger.info(f"fNIRS processing complete (pre-truncation): {num_total_fnirs_frames_processed} total frames " +
                f"({num_actual_fnirs_frames_processed} actual, {num_total_fnirs_frames_processed - num_actual_fnirs_frames_processed} placeholders).")

    if num_actual_fnirs_frames_processed == 0:
        logger.error("No actual fNIRS frames available after gap processing.")
        return None
    if num_total_fnirs_frames_processed < fnirs_frames_per_window:
         logger.error(f"Not enough total fNIRS frames ({num_total_fnirs_frames_processed}) after gap processing.")
         return None

    # --- 1.9: Truncate fNIRS processed data based on ending quality ---
    logger.info("Phase 1.9: Checking fNIRS processed data end quality for truncation...")
    original_length_pre_trunc = num_total_fnirs_frames_processed
    if num_total_fnirs_frames_processed >= drift_chunk_size_frames :
        final_valid_idx = num_total_fnirs_frames_processed - 1
        found_good_end_chunk = False
        for current_potential_end_idx in range(num_total_fnirs_frames_processed - 1, drift_chunk_size_frames - 2, -1):
            test_chunk_start_idx = current_potential_end_idx - drift_chunk_size_frames + 1
            if test_chunk_start_idx < 0: break
            chunk_to_test_actual_flags = fnirs_is_actual_frame[test_chunk_start_idx : current_potential_end_idx + 1]
            if len(chunk_to_test_actual_flags) < drift_chunk_size_frames: continue
            density = np.sum(chunk_to_test_actual_flags) / len(chunk_to_test_actual_flags)
            logger.debug(f"  Checking density for fNIRS processed frames from index {test_chunk_start_idx} to {current_potential_end_idx}: {density*100:.1f}% actual.")
            if density >= min_actual_frame_density_for_truncation:
                final_valid_idx = current_potential_end_idx
                logger.info(f"  Found suitable end for fNIRS processed data at index {final_valid_idx}. Density of last {len(chunk_to_test_actual_flags)} frames: {density*100:.1f}%.")
                found_good_end_chunk = True
                break

        if not found_good_end_chunk:
             logger.warning(f"  No trailing chunk of {drift_chunk_size_frames} frames met {min_actual_frame_density_for_truncation*100}% actual data criteria. Max valid index considered: {final_valid_idx}.")

        if final_valid_idx < (original_length_pre_trunc - 1):
            if (final_valid_idx + 1) < min_frames_for_late_offset_calc:
                 logger.warning(f"fNIRS data would be too short ({final_valid_idx + 1} frames) after quality truncation. Min required: {min_frames_for_late_offset_calc}. No truncation performed.")
            else:
                logger.info(f"Truncating fNIRS processed data to end at index {final_valid_idx}. Original length: {original_length_pre_trunc}.")
                fnirs_ts_ms = fnirs_ts_ms[:final_valid_idx + 1]
                fnirs_data_processed = fnirs_data_processed[:final_valid_idx + 1]
                fnirs_is_actual_frame = fnirs_is_actual_frame[:final_valid_idx + 1]
                num_total_fnirs_frames_processed = len(fnirs_ts_ms)
                num_actual_fnirs_frames_processed = np.sum(fnirs_is_actual_frame)
                logger.info(f"fNIRS data after truncation: {num_total_fnirs_frames_processed} total frames ({num_actual_fnirs_frames_processed} actual).")
        else:
            logger.info("  No truncation needed for fNIRS data based on ending quality.")
    else:
        logger.info(f"  Skipping fNIRS end quality check: total processed fNIRS frames ({num_total_fnirs_frames_processed}) is less than chunk_size ({drift_chunk_size_frames}).")

    # --- 1.95: Adjust EEG Overlap Post fNIRS Truncation ---
    logger.info("Phase 1.95: Adjusting EEG overlap based on final fNIRS stream length...")
    if num_total_fnirs_frames_processed > 0 :
        final_fnirs_stream_end_ts = fnirs_ts_ms[-1]
        new_last_eeg_idx = np.searchsorted(eeg_ts_ms, final_fnirs_stream_end_ts, side='right') - 1

        if new_last_eeg_idx < first_eeg_idx_in_overlap or (new_last_eeg_idx - first_eeg_idx_in_overlap + 1) < eeg_frames_per_window:
            logger.warning(f"Adjusted EEG overlap would result in too few EEG frames (new_last_idx: {new_last_eeg_idx}, first_idx: {first_eeg_idx_in_overlap}). Attempting to use original EEG end if valid.")
            if (original_last_eeg_idx_in_overlap - first_eeg_idx_in_overlap + 1) < eeg_frames_per_window:
                 logger.error("Original EEG overlap also too short for a window. Cannot proceed.")
                 return None
            # Keep last_eeg_idx_in_overlap as original_last_eeg_idx_in_overlap
            logger.info(f"Using original last_eeg_idx_in_overlap: {last_eeg_idx_in_overlap}")
        else:
            if new_last_eeg_idx < last_eeg_idx_in_overlap:
                 logger.info(f"Adjusted last_eeg_idx_in_overlap from {last_eeg_idx_in_overlap} to {new_last_eeg_idx} (new EEG end ts: {eeg_ts_ms[new_last_eeg_idx]:.3f} ms). T_fNIRS_end: {final_fnirs_stream_end_ts:.3f}")
            last_eeg_idx_in_overlap = new_last_eeg_idx

        num_usable_eeg_frames_final = last_eeg_idx_in_overlap - first_eeg_idx_in_overlap + 1
        logger.info(f"Final usable EEG frames in overlap: {num_usable_eeg_frames_final} (indices {first_eeg_idx_in_overlap} to {last_eeg_idx_in_overlap})")
        if num_usable_eeg_frames_final < eeg_frames_per_window:
            logger.error("Not enough EEG frames for a window after fNIRS truncation adjustment.")
            return None
    else:
        logger.error("No fNIRS frames remaining after truncation, cannot adjust EEG overlap or create windows.")
        return None

    if num_actual_fnirs_frames_processed == 0:
        logger.error("No actual fNIRS frames available after potential truncation. Cannot proceed.")
        return None
    if num_total_fnirs_frames_processed < fnirs_frames_per_window:
         logger.error(f"Not enough total fNIRS frames ({num_total_fnirs_frames_processed}) for a window after potential truncation.")
         return None
    if num_total_fnirs_frames_processed < min_frames_for_late_offset_calc:
        logger.warning(f"Not enough total fNIRS frames ({num_total_fnirs_frames_processed}) for late offset calculation after truncation. Drift calculation might be unreliable or use fallback.")

    # --- Phase 2: fNIRS Timing Characterization ---
    logger.info("Phase 2: Characterizing fNIRS timing and drift...")
    avg_early_fnirs_offset = 0.0
    avg_late_fnirs_offset = 0.0

    actual_frame_indices_in_processed = np.where(fnirs_is_actual_frame)[0]
    if len(actual_frame_indices_in_processed) == 0 and num_actual_fnirs_frames_processed > 0:
        logger.error("Mismatch: num_actual_fnirs_frames_processed > 0 but no actual frame indices found.")
        return None
    elif num_actual_fnirs_frames_processed == 0:
        logger.warning("No actual fNIRS frames for offset/drift calculation. Setting defaults.")

    if num_actual_fnirs_frames_processed > 0:
        num_early_chunk_actual_calc = min(drift_chunk_size_frames, num_actual_fnirs_frames_processed)
        if num_early_chunk_actual_calc > 0:
            early_offsets = []
            first_actual_ts_for_early_chunk = -1
            actual_frames_count_in_early_chunk = 0
            for i in range(num_early_chunk_actual_calc):
                processed_idx_of_actual_frame = actual_frame_indices_in_processed[i]
                actual_ts = fnirs_ts_ms[processed_idx_of_actual_frame]
                if first_actual_ts_for_early_chunk < 0: first_actual_ts_for_early_chunk = actual_ts
                ideal_ts_in_chunk = first_actual_ts_for_early_chunk + (actual_frames_count_in_early_chunk * fnirs_ideal_period_ms)
                offset = actual_ts - ideal_ts_in_chunk
                logger.debug(f"  Early chunk - Actual Frame Idx (proc): {processed_idx_of_actual_frame}, Actual TS: {actual_ts:.3f}, Ideal TS: {ideal_ts_in_chunk:.3f}, Offset: {offset:.3f} ms")
                early_offsets.append(offset)
                actual_frames_count_in_early_chunk +=1
            if early_offsets:
                if len(early_offsets) > 1:
                    actual_early_periods = []
                    for j in range(num_early_chunk_actual_calc - 1):
                        idx1, idx2 = actual_frame_indices_in_processed[j], actual_frame_indices_in_processed[j+1]
                        actual_early_periods.append(fnirs_ts_ms[idx2] - fnirs_ts_ms[idx1])
                    if actual_early_periods:
                        logger.info(f"Avg actual period for early chunk (actual frames): {np.mean(actual_early_periods):.3f} ms (based on {len(actual_early_periods)} periods between {num_early_chunk_actual_calc} actual frames)")
                avg_early_fnirs_offset = np.mean(early_offsets)
            logger.info(f"Avg early fNIRS offset (from actuals): {avg_early_fnirs_offset:.3f} ms (based on {len(early_offsets)} actual frames)")
        else:
            logger.warning("Not enough actual frames for early fNIRS offset calculation.")
            avg_early_fnirs_offset = 0.0
    else:
        logger.warning("No actual fNIRS frames in stream for early offset. Defaulting to 0.")
        avg_early_fnirs_offset = 0.0

    if num_total_fnirs_frames_processed >= min_frames_for_late_offset_calc:
        num_late_chunk_calc = drift_chunk_size_frames
        late_offsets_processed = []
        late_chunk_start_idx_processed = num_total_fnirs_frames_processed - num_late_chunk_calc
        first_ts_for_late_chunk_processed = fnirs_ts_ms[late_chunk_start_idx_processed]
        for i in range(num_late_chunk_calc):
            current_processed_idx = late_chunk_start_idx_processed + i
            ts_in_processed_chunk = fnirs_ts_ms[current_processed_idx]
            ideal_ts_in_chunk = first_ts_for_late_chunk_processed + (i * fnirs_ideal_period_ms)
            offset = ts_in_processed_chunk - ideal_ts_in_chunk
            logger.debug(f"  Late chunk (processed) - Idx: {current_processed_idx}, TS: {ts_in_processed_chunk:.3f} (Actual: {fnirs_is_actual_frame[current_processed_idx]}), Ideal TS: {ideal_ts_in_chunk:.3f}, Offset: {offset:.3f} ms")
            late_offsets_processed.append(offset)
        if late_offsets_processed:
            if len(late_offsets_processed) > 1:
                processed_late_periods = []
                for j in range(num_late_chunk_calc - 1):
                    idx1, idx2 = late_chunk_start_idx_processed + j, late_chunk_start_idx_processed + j + 1
                    processed_late_periods.append(fnirs_ts_ms[idx2] - fnirs_ts_ms[idx1])
                if processed_late_periods:
                    logger.info(f"Avg period for late chunk (processed frames): {np.mean(processed_late_periods):.3f} ms (based on {len(processed_late_periods)} periods between {num_late_chunk_calc} processed frames)")
            avg_late_fnirs_offset = np.mean(late_offsets_processed)
        logger.info(f"Avg late fNIRS offset (from processed stream end): {avg_late_fnirs_offset:.3f} ms (based on {len(late_offsets_processed)} frames)")
    else:
        logger.warning(f"Not enough total processed fNIRS frames ({num_total_fnirs_frames_processed}) for late offset. Using early offset as late.")
        avg_late_fnirs_offset = avg_early_fnirs_offset

    if num_actual_fnirs_frames_processed > 0 and num_total_fnirs_frames_processed > 0 :
        ts_first_overall_actual_fnirs = fnirs_ts_ms[actual_frame_indices_in_processed[0]]
        ts_last_frame_in_processed_stream = fnirs_ts_ms[-1]
        time_delta_for_drift = ts_last_frame_in_processed_stream - ts_first_overall_actual_fnirs
        if time_delta_for_drift > fnirs_ideal_period_ms:
            drift_ms_per_ms = (avg_late_fnirs_offset - avg_early_fnirs_offset) / time_delta_for_drift
            drift_ms_per_s = drift_ms_per_ms * 1000.0
            logger.info(f"Calculated fNIRS timing drift: {drift_ms_per_s:.3f} ms/s " +
                        f"(early_offset: {avg_early_fnirs_offset:.3f}, late_offset_processed: {avg_late_fnirs_offset:.3f} over {time_delta_for_drift:.3f} ms span)")
        else:
            logger.info(f"fNIRS timing drift calculation skipped: time delta ({time_delta_for_drift:.3f}ms) not sufficiently positive.")
    else:
        logger.info("fNIRS timing drift calculation skipped: no actual or total processed frames.")

    if len(actual_frame_indices_in_processed) == 0:
        logger.error("Cannot project first window start: no actual fNIRS frames remain.")
        return None
    first_actual_fnirs_ts_in_processed = fnirs_ts_ms[actual_frame_indices_in_processed[0]]
    projected_first_window_start_ts = first_actual_fnirs_ts_in_processed - avg_early_fnirs_offset
    logger.info(f"Timestamp of first actual fNIRS frame in (final) processed data: {first_actual_fnirs_ts_in_processed:.3f} ms")
    logger.info(f"Projected start timestamp for the first window: {projected_first_window_start_ts:.3f} ms")

    # --- Phase 3: Align EEG and Iterate to Create Windows ---
    logger.info("Phase 3: Aligning EEG and iterating to create windows...")
    relative_eeg_idx_offset = np.searchsorted(eeg_ts_ms[first_eeg_idx_in_overlap : last_eeg_idx_in_overlap + 1],
                                   projected_first_window_start_ts, side='left')
    adjusted_first_eeg_global_idx = first_eeg_idx_in_overlap + relative_eeg_idx_offset

    if adjusted_first_eeg_global_idx > last_eeg_idx_in_overlap:
        logger.error("No suitable EEG frame found at or after projected_first_window_start_ts within the usable EEG range.")
        return None
    logger.info(f"Adjusted first EEG window global index: {adjusted_first_eeg_global_idx} " +
                f"(timestamp: {eeg_ts_ms[adjusted_first_eeg_global_idx]:.3f} ms)")

    windows_list: List[Dict[str, Union[np.ndarray, torch.Tensor, Dict]]] = []
    current_eeg_start_global_idx = adjusted_first_eeg_global_idx
    current_fnirs_processed_idx = 0
    window_count = 0

    logger.info(f"Starting window creation loop. EEG from global idx {current_eeg_start_global_idx} (up to {last_eeg_idx_in_overlap}), fNIRS from processed idx {current_fnirs_processed_idx} (up to {num_total_fnirs_frames_processed -1}).")

    while True:
        eeg_chunk_end_global_idx = current_eeg_start_global_idx + eeg_frames_per_window - 1
        fnirs_chunk_end_processed_idx = current_fnirs_processed_idx

        if eeg_chunk_end_global_idx > last_eeg_idx_in_overlap:
            logger.info(f"Stopping: Not enough usable EEG frames for window {window_count} (needed up to {eeg_chunk_end_global_idx}, max is {last_eeg_idx_in_overlap}).")
            break
        if fnirs_chunk_end_processed_idx >= num_total_fnirs_frames_processed:
            logger.info(f"Stopping: Not enough processed fNIRS frames for window {window_count} (needed index {fnirs_chunk_end_processed_idx}, available {num_total_fnirs_frames_processed}).")
            break

        eeg_w_np = eeg_data[current_eeg_start_global_idx : current_eeg_start_global_idx + eeg_frames_per_window]
        fnirs_w_np = fnirs_data_processed[current_fnirs_processed_idx : current_fnirs_processed_idx + fnirs_frames_per_window]
        eeg_ts_window_ms = eeg_ts_ms[current_eeg_start_global_idx : current_eeg_start_global_idx + eeg_frames_per_window]
        fnirs_ts_window_ms = fnirs_ts_ms[current_fnirs_processed_idx : current_fnirs_processed_idx + fnirs_frames_per_window]
        window_ideal_fnirs_event_ts = projected_first_window_start_ts + (window_count * fnirs_ideal_period_ms)
        actual_this_fnirs_event_ts = fnirs_ts_window_ms[0]
        is_actual_fnirs = fnirs_is_actual_frame[current_fnirs_processed_idx]

        metadata = {
            'window_idx': window_count,
            'eeg_frame_indices_abs': (current_eeg_start_global_idx, eeg_chunk_end_global_idx),
            'fnirs_frame_indices_processed': (current_fnirs_processed_idx, fnirs_chunk_end_processed_idx),
            'eeg_timestamps_ms': eeg_ts_window_ms, 'fnirs_timestamps_ms': fnirs_ts_window_ms,
            'fnirs_frame_is_actual': is_actual_fnirs,
            'ideal_fnirs_event_ts_projected': window_ideal_fnirs_event_ts,
            'actual_fnirs_event_ts': actual_this_fnirs_event_ts,
            'fnirs_event_to_ideal_offset_ms': actual_this_fnirs_event_ts - window_ideal_fnirs_event_ts,
            'source_projected_first_actual_fnirs_start_ts': projected_first_window_start_ts,
            'source_avg_early_fnirs_offset_calc': avg_early_fnirs_offset,
            'source_avg_late_fnirs_offset_calc': avg_late_fnirs_offset,
            'notes': "Window aligned by projected fNIRS rhythm. fNIRS data is zeros if 'fnirs_frame_is_actual' is False."
        }
        windows_list.append({'eeg': eeg_w_np, 'fnirs': fnirs_w_np, 'metadata': metadata})
        current_eeg_start_global_idx += eeg_frames_per_window
        current_fnirs_processed_idx += fnirs_frames_per_window
        window_count += 1

    if not windows_list: logger.warning("No windows were created.")
    else: logger.info(f"Successfully created {len(windows_list)} windows.")

    # --- Phase 4: Prepare for NumPy Array Output (New Logic) ---
    logger.info("Phase 4: Transforming windows_list to NumPy array outputs.")

    # 4.1 Determine Master Timeline Based on Raw Full Extent
    raw_eeg_ts_ms = eeg_timestamps[:, 1].astype(float) # Original, full eeg timestamps
    raw_fnirs_ts_ms = fnirs_timestamps[:, 1].astype(float) # Original, full fnirs timestamps

    eeg_single_frame_duration_ms = fnirs_ideal_period_ms / eeg_frames_per_window

    if len(raw_eeg_ts_ms) == 0 and len(raw_fnirs_ts_ms) == 0:
        logger.error("Both raw EEG and fNIRS timestamps are empty. Cannot create master timeline.")
        return None # Or return empty tuple structure
    elif len(raw_eeg_ts_ms) == 0:
        overall_min_ts = raw_fnirs_ts_ms[0]
        overall_max_ts = raw_fnirs_ts_ms[-1] + fnirs_ideal_period_ms
    elif len(raw_fnirs_ts_ms) == 0:
        overall_min_ts = raw_eeg_ts_ms[0]
        overall_max_ts = raw_eeg_ts_ms[-1] + eeg_single_frame_duration_ms
    else:
        overall_min_ts = min(raw_eeg_ts_ms[0], raw_fnirs_ts_ms[0])
        overall_max_ts = max(raw_eeg_ts_ms[-1] + eeg_single_frame_duration_ms,
                             raw_fnirs_ts_ms[-1] + fnirs_ideal_period_ms)

    if overall_max_ts <= overall_min_ts:
        logger.warning(f"Overall master timeline duration is zero or negative ({overall_min_ts=} to {overall_max_ts=}).")
        total_windows = 0
    else:
        total_windows = int(np.ceil((overall_max_ts - overall_min_ts) / fnirs_ideal_period_ms))

    if total_windows == 0:
         logger.warning("Master timeline results in 0 total_windows. All output arrays will be empty.")

    main_clock_timestamps_array = overall_min_ts + np.arange(total_windows) * fnirs_ideal_period_ms

    # 4.2 Initialize Output NumPy Arrays
    num_eeg_chans = eeg_data.shape[1]
    # Ensure eeg_data.shape[2] exists and is valid, e.g. 15 for 7*15=105
    eeg_samples_per_raw_frame = eeg_data.shape[2] if eeg_data.ndim == 3 and eeg_data.shape[2] > 0 else 15 # Default if not 3D somehow
    eeg_samples_per_output_window = eeg_frames_per_window * eeg_samples_per_raw_frame
    num_fnirs_chans = fnirs_data.shape[1]
    fnirs_samples_per_raw_frame = fnirs_data.shape[2] if fnirs_data.ndim == 3 and fnirs_data.shape[2] > 0 else 1 # Default

    eeg_windows_data_arr = np.zeros((total_windows, num_eeg_chans, eeg_samples_per_output_window), dtype=eeg_data.dtype)
    fnirs_windows_data_arr = np.zeros((total_windows, num_fnirs_chans, fnirs_samples_per_raw_frame), dtype=fnirs_data.dtype)
    eeg_validity_mask_arr = np.zeros(total_windows, dtype=bool)
    fnirs_validity_mask_arr = np.zeros(total_windows, dtype=bool)
    real_eeg_timestamps_arr = np.full(total_windows, np.nan, dtype=float)
    real_fnirs_timestamps_arr = np.full(total_windows, np.nan, dtype=float)

    # 4.3 Map windows_list to the NumPy Arrays
    if windows_list:
        for original_window in windows_list:
            map_key_ts = original_window['metadata'].get('ideal_fnirs_event_ts_projected',
                                                         original_window['metadata'].get('actual_fnirs_event_ts'))
            if map_key_ts is None:
                logger.warning(f"Skipping original window (idx {original_window['metadata']['window_idx']}) due to missing mapping timestamp.")
                continue

            if total_windows > 0:
                diffs = np.abs(main_clock_timestamps_array - map_key_ts)
                slot_idx = np.argmin(diffs)
                if diffs[slot_idx] > (fnirs_ideal_period_ms / 1.9): # Relaxed tolerance for mapping
                    logger.debug(f"Original window (ideal_ts {map_key_ts:.2f}) too far from any master clock slot. Min diff: {diffs[slot_idx]:.2f}. Target slot: {main_clock_timestamps_array[slot_idx]:.2f}. Skipping.")
                    continue
            else:
                continue

            # EEG Data
            eeg_from_list = original_window['eeg']
            if eeg_from_list.shape == (eeg_frames_per_window, num_eeg_chans, eeg_samples_per_raw_frame):
                eeg_windows_data_arr[slot_idx] = eeg_from_list.transpose(1,0,2).reshape(num_eeg_chans, eeg_samples_per_output_window)
                eeg_validity_mask_arr[slot_idx] = True
                real_eeg_timestamps_arr[slot_idx] = original_window['metadata']['eeg_timestamps_ms'][0]
            else:
                logger.warning(f"EEG data in original_window {original_window['metadata']['window_idx']} has unexpected shape {eeg_from_list.shape}. Expected {(eeg_frames_per_window, num_eeg_chans, eeg_samples_per_raw_frame)}. Skipping EEG for this slot.")

            # fNIRS Data
            if original_window['metadata']['fnirs_frame_is_actual']:
                fnirs_from_list = original_window['fnirs']
                if fnirs_from_list.shape == (fnirs_frames_per_window, num_fnirs_chans, fnirs_samples_per_raw_frame):
                    fnirs_windows_data_arr[slot_idx] = fnirs_from_list.reshape(num_fnirs_chans, fnirs_samples_per_raw_frame) # Reshape for (chans, samples) like format
                    fnirs_validity_mask_arr[slot_idx] = True
                    real_fnirs_timestamps_arr[slot_idx] = original_window['metadata']['actual_fnirs_event_ts']
                else:
                    logger.warning(f"Actual fNIRS data in original_window {original_window['metadata']['window_idx']} has unexpected shape {fnirs_from_list.shape}. Expected {(fnirs_frames_per_window, num_fnirs_chans, fnirs_samples_per_raw_frame)}. Skipping fNIRS for this slot.")
    else: # No windows in windows_list
        logger.info("windows_list is empty. Output arrays will reflect no mapped data.")

    # 4.4 Calculate Scalar Indices and Metadata
    def get_span_indices(mask_arr):
        valid_indices = np.where(mask_arr)[0]
        if len(valid_indices) > 0:
            return valid_indices[0], valid_indices[-1] + 1 # Exclusive end
        return -1, -1

    eeg_start_idx, eeg_end_idx = get_span_indices(eeg_validity_mask_arr)
    fnirs_start_idx, fnirs_end_idx = get_span_indices(fnirs_validity_mask_arr)

    both_valid_mask = eeg_validity_mask_arr & fnirs_validity_mask_arr
    both_start_idx, both_end_idx = get_span_indices(both_valid_mask)

    # Percentage Metadata
    percent_eeg_trimmed = 100.0
    percent_fnirs_trimmed = 100.0
    percent_fnirs_missing = 0.0 # Default if no valid fNIRS span

    if len(raw_eeg_ts_ms) > 0:
        original_eeg_duration = (raw_eeg_ts_ms[-1] + eeg_single_frame_duration_ms) - raw_eeg_ts_ms[0]
        if eeg_start_idx != -1 and eeg_end_idx > eeg_start_idx: # Check for valid span
            # Find the actual eeg data coverage using real_eeg_timestamps_arr within the valid span
            valid_eeg_actual_ts = real_eeg_timestamps_arr[eeg_start_idx:eeg_end_idx]
            valid_eeg_actual_ts = valid_eeg_actual_ts[~np.isnan(valid_eeg_actual_ts)] # Remove NaNs
            if len(valid_eeg_actual_ts) > 0:
                actual_eeg_coverage_start = np.min(valid_eeg_actual_ts)
                # Find the metadata of the original window that contributed to the last valid eeg slot
                last_eeg_slot_original_ts_end = np.nan
                if windows_list and total_windows > 0:
                    for win_in_list in reversed(windows_list):
                        map_key_ts = win_in_list['metadata'].get('ideal_fnirs_event_ts_projected', win_in_list['metadata'].get('actual_fnirs_event_ts'))
                        if map_key_ts is None: continue
                        _mapped_slot_idx = np.argmin(np.abs(main_clock_timestamps_array - map_key_ts))
                        if _mapped_slot_idx == (eeg_end_idx - 1) and eeg_validity_mask_arr[_mapped_slot_idx]:
                            last_eeg_slot_original_ts_end = win_in_list['metadata']['eeg_timestamps_ms'][-1] + eeg_single_frame_duration_ms
                            break
                if not np.isnan(last_eeg_slot_original_ts_end):
                    actual_eeg_duration_covered = last_eeg_slot_original_ts_end - actual_eeg_coverage_start
                    if original_eeg_duration > 1e-6: # Avoid division by zero
                        percent_eeg_trimmed = max(0.0, (1.0 - (actual_eeg_duration_covered / original_eeg_duration)) * 100.0)
                    else: percent_eeg_trimmed = 0.0 if actual_eeg_duration_covered > 0 else 100.0
                else: percent_eeg_trimmed = 100.0 # Fallback if end couldn't be determined
            else: percent_eeg_trimmed = 100.0 # No valid actual EEG timestamps
        # else: percent_eeg_trimmed remains 100.0 if no valid span
    # else: percent_eeg_trimmed remains 100.0 if no raw_eeg_ts_ms

    if len(raw_fnirs_ts_ms) > 0:
        original_fnirs_duration = (raw_fnirs_ts_ms[-1] + fnirs_ideal_period_ms) - raw_fnirs_ts_ms[0]
        if fnirs_start_idx != -1 and fnirs_end_idx > fnirs_start_idx:
            valid_fnirs_actual_ts = real_fnirs_timestamps_arr[fnirs_start_idx:fnirs_end_idx]
            valid_fnirs_actual_ts = valid_fnirs_actual_ts[~np.isnan(valid_fnirs_actual_ts)]
            if len(valid_fnirs_actual_ts) > 0:
                actual_fnirs_coverage_start = np.min(valid_fnirs_actual_ts)
                last_fnirs_slot_original_ts_end = np.nan
                if windows_list and total_windows > 0:
                    for win_in_list in reversed(windows_list):
                        map_key_ts = win_in_list['metadata'].get('ideal_fnirs_event_ts_projected', win_in_list['metadata'].get('actual_fnirs_event_ts'))
                        if map_key_ts is None: continue
                        _mapped_slot_idx = np.argmin(np.abs(main_clock_timestamps_array - map_key_ts))
                        if _mapped_slot_idx == (fnirs_end_idx - 1) and fnirs_validity_mask_arr[_mapped_slot_idx]:
                            last_fnirs_slot_original_ts_end = win_in_list['metadata']['actual_fnirs_event_ts'] + fnirs_ideal_period_ms
                            break
                if not np.isnan(last_fnirs_slot_original_ts_end):
                    actual_fnirs_duration_covered = last_fnirs_slot_original_ts_end - actual_fnirs_coverage_start
                    if original_fnirs_duration > 1e-6:
                        percent_fnirs_trimmed = max(0.0, (1.0 - (actual_fnirs_duration_covered / original_fnirs_duration)) * 100.0)
                    else: percent_fnirs_trimmed = 0.0 if actual_fnirs_duration_covered > 0 else 100.0
                else: percent_fnirs_trimmed = 100.0
            else: percent_fnirs_trimmed = 100.0
        # else: percent_fnirs_trimmed remains 100.0
    # else: percent_fnirs_trimmed remains 100.0

    if fnirs_start_idx != -1 and fnirs_end_idx > fnirs_start_idx:
        expected_fnirs_in_span = fnirs_end_idx - fnirs_start_idx
        actual_fnirs_in_span = np.sum(fnirs_validity_mask_arr[fnirs_start_idx:fnirs_end_idx])
        if expected_fnirs_in_span > 0:
            percent_fnirs_missing = max(0.0, ((expected_fnirs_in_span - actual_fnirs_in_span) / float(expected_fnirs_in_span)) * 100.0)
    # else: percent_fnirs_missing remains 0.0

    # Clamp percentages to [0, 100]
    percent_eeg_trimmed = max(0.0, min(100.0, percent_eeg_trimmed))
    percent_fnirs_trimmed = max(0.0, min(100.0, percent_fnirs_trimmed))
    percent_fnirs_missing = max(0.0, min(100.0, percent_fnirs_missing))

    eeg_start_idx = int(eeg_start_idx)
    eeg_end_idx = int(eeg_end_idx)
    fnirs_start_idx = int(fnirs_start_idx)
    fnirs_end_idx = int(fnirs_end_idx)
    both_start_idx = int(both_start_idx)
    both_end_idx = int(both_end_idx)

    logger.info(f"NumPy array creation complete. Total master windows: {total_windows}")
    logger.info(f"EEG: {np.sum(eeg_validity_mask_arr)} valid windows. Span: {eeg_start_idx}-{eeg_end_idx}. Trimmed: {percent_eeg_trimmed:.1f}%")
    logger.info(f"fNIRS: {np.sum(fnirs_validity_mask_arr)} valid windows. Span: {fnirs_start_idx}-{fnirs_end_idx}. Trimmed: {percent_fnirs_trimmed:.1f}%. Missing in span: {percent_fnirs_missing:.1f}%")
    logger.info(f"Both: Valid span: {both_start_idx}-{both_end_idx}")

    # --- Phase 5: Instantiate WindowDataset and Return ---
    logger.info("Phase 5: Instantiating WindowDataset and preparing final output.")

    summary_metadata = {
        'percent_eeg_trimmed': percent_eeg_trimmed,
        'percent_fnirs_trimmed': percent_fnirs_trimmed,
        'percent_fnirs_missing_in_span': percent_fnirs_missing,
        'total_master_windows_in_input_arrays': total_windows,
        'eeg_master_span_in_input_arrays': (eeg_start_idx, eeg_end_idx),
        'fnirs_master_span_in_input_arrays': (fnirs_start_idx, fnirs_end_idx),
        'both_master_span_in_input_arrays': (both_start_idx, both_end_idx)
    }
    
    # Pass through retained_fnirs_indices if they exist in metadata
    if metadata and 'retained_fnirs_indices' in metadata:
        summary_metadata['retained_fnirs_indices'] = metadata['retained_fnirs_indices']

    dataset_instance: Optional[WindowDataset] = None
    if total_windows == 0 : # If master timeline was empty
        logger.warning("Total master windows is 0, WindowDataset cannot be meaningfully created with current inputs.")
        # Fall through, dataset_instance will be None
    elif not (isinstance(both_start_idx, int) and isinstance(both_end_idx, int) and \
            both_start_idx != -1 and both_end_idx != -1 and both_start_idx < both_end_idx):
        logger.warning(f"Cannot create WindowDataset: No valid span where both modalities are present. "
                       f"both_start_idx: {both_start_idx}, both_end_idx: {both_end_idx}")
        # Fall through, dataset_instance will be None
    else:
        try:
            dataset_instance = WindowDataset(
                eeg_windows_data_arr=eeg_windows_data_arr,
                fnirs_windows_data_arr=fnirs_windows_data_arr,
                eeg_validity_mask_arr=eeg_validity_mask_arr,
                fnirs_validity_mask_arr=fnirs_validity_mask_arr,
                real_eeg_timestamps_arr=real_eeg_timestamps_arr,
                real_fnirs_timestamps_arr=real_fnirs_timestamps_arr,
                main_clock_timestamps_array=main_clock_timestamps_array,
                eeg_start_idx=eeg_start_idx, # master eeg_start_idx
                fnirs_start_idx=fnirs_start_idx, # master fnirs_start_idx
                eeg_end_idx=eeg_end_idx, # master eeg_end_idx
                fnirs_end_idx=fnirs_end_idx, # master fnirs_end_idx
                both_start_idx=both_start_idx, # the slice_start for dataset
                both_end_idx=both_end_idx, # the slice_end for dataset
                return_torch_tensors=return_torch_tensors
            )
            logger.info(f"Successfully created WindowDataset with {len(dataset_instance)} synchronized windows.")
        except ValueError as ve:
            logger.error(f"Error instantiating WindowDataset: {ve}. This typically means no valid 'both' span.")
            dataset_instance = None # Ensure it's None on failure
        except Exception as e:
            logger.error(f"Unexpected error during WindowDataset instantiation: {e}", exc_info=True)
            dataset_instance = None # Ensure it's None

    logger.info(f"Finalizing create_windows. summary_metadata keys: {list(summary_metadata.keys())}")

    return dataset_instance, summary_metadata


def create_windows_eeg_only(
    eeg_data: Optional[np.ndarray] = None,
    eeg_timestamps: Optional[np.ndarray] = None,
    return_torch_tensors: bool = False,
    # TODO: Consider making these parameters or module-level constants
    eeg_frames_per_window: int = 7,
    ideal_period_ms: float = 210.0,  # Using fnirs_ideal_period_ms as a base for window rate
    metadata: Optional[dict] = None
):
    """
    Creates time-aligned windows from EEG data only.
    Returns a WindowDataset and summary metadata.
    """
    logger.info("Starting create_windows_eeg_only...")

    # --- 1. Validate Inputs ---
    if not (eeg_data is not None and len(eeg_data) > 0 and
            eeg_timestamps is not None and len(eeg_timestamps) > 0):
        logger.error("EEG data or timestamps are missing or empty.")
        return None, {"error": "EEG data or timestamps missing"}
    if eeg_data.ndim != 3 or eeg_data.shape[1] == 0 or eeg_data.shape[2] == 0:
        logger.error(f"EEG data has unexpected shape: {eeg_data.shape}.")
        return None, {"error": f"EEG data unexpected shape: {eeg_data.shape}"}

    try:
        eeg_ts_ms = eeg_timestamps[:, 1].astype(float)
        if eeg_ts_ms.shape[0] != eeg_data.shape[0]:
            logger.error(f"EEG timestamp count ({eeg_ts_ms.shape[0]}) " +
                           f"doesn't match EEG data frame count ({eeg_data.shape[0]}).")
            return None, {"error": "EEG timestamp count mismatch"}
    except IndexError:
        logger.error("EEG timestamps array does not have a second column (index 1).")
        return None, {"error": "EEG timestamps wrong format"}
    except Exception as e:
        logger.error(f"Could not extract or validate EEG timestamps: {e}.")
        return None, {"error": f"EEG timestamp extraction error: {e}"}

    logger.info(f"Successfully extracted {len(eeg_ts_ms)} EEG timestamps.")

    if len(eeg_ts_ms) < eeg_frames_per_window:
        logger.error("Not enough EEG frames for a single window.")
        return None, {"error": "Not enough EEG frames for a window"}

    first_eeg_idx_in_overlap = 0
    last_eeg_idx_in_overlap = len(eeg_ts_ms) - 1
    
    # --- Phase 3: Iterate to Create Windows (Simplified for EEG-only) ---
    logger.info("Phase 3: Iterating to create EEG windows...")
    
    windows_list: List[Dict[str, Union[np.ndarray, Dict]]] = []
    current_eeg_start_global_idx = first_eeg_idx_in_overlap
    window_count = 0
    
    projected_first_window_start_ts = eeg_ts_ms[0] # Anchor on the first EEG timestamp

    logger.info(f"Starting window creation loop. EEG from global idx {current_eeg_start_global_idx} (up to {last_eeg_idx_in_overlap}).")

    while True:
        eeg_chunk_end_global_idx = current_eeg_start_global_idx + eeg_frames_per_window - 1

        if eeg_chunk_end_global_idx > last_eeg_idx_in_overlap:
            logger.info(f"Stopping: Not enough usable EEG frames for window {window_count} (needed up to {eeg_chunk_end_global_idx}, max is {last_eeg_idx_in_overlap}).")
            break

        eeg_w_np = eeg_data[current_eeg_start_global_idx : current_eeg_start_global_idx + eeg_frames_per_window]
        eeg_ts_window_ms = eeg_ts_ms[current_eeg_start_global_idx : current_eeg_start_global_idx + eeg_frames_per_window]
        
        window_ideal_event_ts = projected_first_window_start_ts + (window_count * ideal_period_ms)

        metadata = {
            'window_idx': window_count,
            'eeg_frame_indices_abs': (current_eeg_start_global_idx, eeg_chunk_end_global_idx),
            'eeg_timestamps_ms': eeg_ts_window_ms,
            'ideal_event_ts_projected': window_ideal_event_ts,
            'actual_event_ts': eeg_ts_window_ms[0], # Use first EEG timestamp in window as actual
            'notes': "EEG-only window."
        }
        windows_list.append({'eeg': eeg_w_np, 'metadata': metadata})
        
        # Advance by one full window's worth of ideal period to find next EEG chunk
        # This assumes EEG data is dense enough. A more robust approach might find the
        # EEG frame closest to (current_event_ts + ideal_period_ms).
        # For now, simple contiguous windowing.
        current_eeg_start_global_idx += eeg_frames_per_window 
        window_count += 1

    if not windows_list: logger.warning("No EEG windows were created.")
    else: logger.info(f"Successfully created {len(windows_list)} EEG windows.")

    # --- Phase 4: Prepare for NumPy Array Output ---
    logger.info("Phase 4: Transforming EEG windows_list to NumPy array outputs.")
    
    raw_eeg_ts_ms = eeg_ts_ms # Already extracted and validated

    eeg_single_frame_duration_ms = ideal_period_ms / eeg_frames_per_window

    overall_min_ts = raw_eeg_ts_ms[0]
    overall_max_ts = raw_eeg_ts_ms[-1] + eeg_single_frame_duration_ms
    
    if overall_max_ts <= overall_min_ts:
        logger.warning(f"Overall master timeline duration is zero or negative ({overall_min_ts=} to {overall_max_ts=}).")
        total_windows = 0
    else:
        total_windows = int(np.ceil((overall_max_ts - overall_min_ts) / ideal_period_ms))

    if total_windows == 0:
         logger.warning("Master timeline results in 0 total_windows. All output arrays will be empty.")

    main_clock_timestamps_array = overall_min_ts + np.arange(total_windows) * ideal_period_ms
    
    num_eeg_chans = eeg_data.shape[1]
    eeg_samples_per_raw_frame = eeg_data.shape[2]
    eeg_samples_per_output_window = eeg_frames_per_window * eeg_samples_per_raw_frame

    eeg_windows_data_arr = np.zeros((total_windows, num_eeg_chans, eeg_samples_per_output_window), dtype=eeg_data.dtype)
    eeg_validity_mask_arr = np.zeros(total_windows, dtype=bool)
    real_eeg_timestamps_arr = np.full(total_windows, np.nan, dtype=float)

    # Empty fNIRS arrays
    # Assuming fnirs_data shape if it were present (e.g. from original create_windows)
    # These need to be defined for WindowDataset, even if empty/dummy
    num_fnirs_chans_dummy = 1 # Placeholder
    fnirs_samples_per_raw_frame_dummy = 1 # Placeholder
    fnirs_windows_data_arr = np.zeros((total_windows, num_fnirs_chans_dummy, fnirs_samples_per_raw_frame_dummy), dtype=np.float32) # Placeholder dtype
    fnirs_validity_mask_arr = np.zeros(total_windows, dtype=bool)
    real_fnirs_timestamps_arr = np.full(total_windows, np.nan, dtype=float)

    if windows_list:
        for original_window in windows_list:
            map_key_ts = original_window['metadata'].get('ideal_event_ts_projected', original_window['metadata']['actual_event_ts'])
            if map_key_ts is None:
                logger.warning(f"Skipping original window (idx {original_window['metadata']['window_idx']}) due to missing mapping timestamp.")
                continue

            if total_windows > 0:
                diffs = np.abs(main_clock_timestamps_array - map_key_ts)
                slot_idx = np.argmin(diffs)
                if diffs[slot_idx] > (ideal_period_ms / 1.9):
                    logger.debug(f"Original EEG window (ideal_ts {map_key_ts:.2f}) too far from any master clock slot. Min diff: {diffs[slot_idx]:.2f}. Skipping.")
                    continue
            else:
                continue
            
            eeg_from_list = original_window['eeg']
            if eeg_from_list.shape == (eeg_frames_per_window, num_eeg_chans, eeg_samples_per_raw_frame):
                eeg_windows_data_arr[slot_idx] = eeg_from_list.transpose(1,0,2).reshape(num_eeg_chans, eeg_samples_per_output_window)
                eeg_validity_mask_arr[slot_idx] = True
                real_eeg_timestamps_arr[slot_idx] = original_window['metadata']['eeg_timestamps_ms'][0]
            else:
                logger.warning(f"EEG data in original_window {original_window['metadata']['window_idx']} has unexpected shape {eeg_from_list.shape}. Skipping EEG for this slot.")
    else:
        logger.info("EEG windows_list is empty. Output arrays will reflect no mapped data.")

    def get_span_indices(mask_arr): # Redefined locally
        valid_indices = np.where(mask_arr)[0]
        if len(valid_indices) > 0:
            return valid_indices[0], valid_indices[-1] + 1 # Exclusive end
        return -1, -1

    eeg_start_idx, eeg_end_idx = get_span_indices(eeg_validity_mask_arr)
    if eeg_start_idx != -1: eeg_start_idx = int(eeg_start_idx)
    if eeg_end_idx != -1: eeg_end_idx = int(eeg_end_idx)
    fnirs_start_idx, fnirs_end_idx = -1, -1 # No fNIRS data

    # For EEG-only, "both" means EEG
    both_valid_mask = eeg_validity_mask_arr
    both_start_idx, both_end_idx = get_span_indices(both_valid_mask)
    if both_start_idx != -1: both_start_idx = int(both_start_idx)
    if both_end_idx != -1: both_end_idx = int(both_end_idx)
    
    percent_eeg_trimmed = 0.0 # Simplified for EEG-only; can be enhanced
    if len(raw_eeg_ts_ms) > 0 and total_windows > 0 and eeg_start_idx != -1:
        # Basic calculation, could be more precise like in original create_windows
        original_duration = raw_eeg_ts_ms[-1] - raw_eeg_ts_ms[0]
        covered_duration = main_clock_timestamps_array[eeg_end_idx-1] - main_clock_timestamps_array[eeg_start_idx] if eeg_end_idx > eeg_start_idx else 0
        if original_duration > 1e-6:
            percent_eeg_trimmed = max(0.0, (1.0 - (covered_duration / original_duration)) * 100.0)
        else:
            percent_eeg_trimmed = 0.0 if covered_duration > 0 else 100.0
    percent_eeg_trimmed = max(0.0, min(100.0, percent_eeg_trimmed))


    summary_metadata = {
        'percent_eeg_trimmed': percent_eeg_trimmed,
        'percent_fnirs_trimmed': 100.0, # No fNIRS
        'percent_fnirs_missing_in_span': 100.0, # No fNIRS
        'total_master_windows_in_input_arrays': total_windows,
        'eeg_master_span_in_input_arrays': (eeg_start_idx, eeg_end_idx),
        'fnirs_master_span_in_input_arrays': (fnirs_start_idx, fnirs_end_idx),
        'both_master_span_in_input_arrays': (both_start_idx, both_end_idx) # 'both' is EEG here
    }

    logger.info(f"NumPy array creation complete for EEG-only. Total master windows: {total_windows}")
    logger.info(f"EEG: {np.sum(eeg_validity_mask_arr)} valid windows. Span: {eeg_start_idx}-{eeg_end_idx}. Trimmed: {percent_eeg_trimmed:.1f}%")

    # --- Phase 5: Instantiate WindowDataset and Return ---
    logger.info("Phase 5: Instantiating WindowDataset for EEG-only.")
    
    dataset_instance: Optional[WindowDataset] = None
    if total_windows == 0:
        logger.warning("Total master windows is 0, WindowDataset cannot be meaningfully created.")
    elif not (isinstance(both_start_idx, int) and isinstance(both_end_idx, int) and \
            both_start_idx != -1 and both_end_idx != -1 and both_start_idx < both_end_idx):
        logger.warning(f"Cannot create WindowDataset: No valid span for EEG data. "
                       f"both_start_idx (eeg_start_idx): {both_start_idx}, both_end_idx (eeg_end_idx): {both_end_idx}")
    else:
        try:
            dataset_instance = WindowDataset(
                eeg_windows_data_arr=eeg_windows_data_arr,
                fnirs_windows_data_arr=fnirs_windows_data_arr, # Empty/zero
                eeg_validity_mask_arr=eeg_validity_mask_arr,
                fnirs_validity_mask_arr=fnirs_validity_mask_arr, # All False
                real_eeg_timestamps_arr=real_eeg_timestamps_arr,
                real_fnirs_timestamps_arr=real_fnirs_timestamps_arr, # All NaN
                main_clock_timestamps_array=main_clock_timestamps_array,
                eeg_start_idx=eeg_start_idx,
                fnirs_start_idx=fnirs_start_idx, # -1
                eeg_end_idx=eeg_end_idx,
                fnirs_end_idx=fnirs_end_idx, # -1
                both_start_idx=both_start_idx, # This is eeg_start_idx
                both_end_idx=both_end_idx,   # This is eeg_end_idx
                return_torch_tensors=return_torch_tensors
            )
            logger.info(f"Successfully created WindowDataset with {len(dataset_instance)} EEG-only windows.")
        except ValueError as ve: # Assuming WindowDataset might raise ValueError for invalid spans
            logger.error(f"Error instantiating WindowDataset for EEG-only: {ve}")
            dataset_instance = None
        except Exception as e:
            logger.error(f"Unexpected error during WindowDataset instantiation for EEG-only: {e}", exc_info=True)
            dataset_instance = None

    logger.info(f"Finalizing create_windows_eeg_only. summary_metadata keys: {list(summary_metadata.keys())}")
    return dataset_instance, summary_metadata


def create_windows_fnirs_only(
    fnirs_data: Optional[np.ndarray] = None,
    fnirs_timestamps: Optional[np.ndarray] = None,
    return_torch_tensors: bool = False,
    # Constants from original create_windows, consider refactoring
    fnirs_frames_per_window: int = 1,
    fnirs_ideal_period_ms: float = 210.0,
    drift_chunk_size_frames: int = 20,
    min_actual_frame_density_for_truncation: float = 0.80,
    min_frames_for_late_offset_calc: int = 20,
    metadata: Optional[dict] = None
):
    """
    Creates time-aligned windows from fNIRS data only.
    Includes gap filling, truncation, and timing characterization for fNIRS.
    Returns a WindowDataset and summary metadata.
    """
    logger.info("Starting create_windows_fnirs_only...")

    # --- 1. Validate Inputs and Extract Timestamps ---
    if not (fnirs_data is not None and len(fnirs_data) > 0 and
            fnirs_timestamps is not None and len(fnirs_timestamps) > 0):
        logger.error("fNIRS data or timestamps are missing or empty.")
        return None, {"error": "fNIRS data or timestamps missing"}
    if fnirs_data.ndim != 3 or fnirs_data.shape[1] == 0 or fnirs_data.shape[2] != 1: # Assuming fnirs_data.shape[2] == 1
        logger.error(f"fNIRS data has unexpected shape: {fnirs_data.shape}.")
        return None, {"error": f"fNIRS data unexpected shape: {fnirs_data.shape}"}

    try:
        fnirs_ts_ms_orig = fnirs_timestamps[:, 1].astype(float)
        if fnirs_ts_ms_orig.shape[0] != fnirs_data.shape[0]:
            logger.error(f"fNIRS timestamp count ({fnirs_ts_ms_orig.shape[0]}) " +
                           f"doesn't match fNIRS data frame count ({fnirs_data.shape[0]}).")
            return None, {"error": "fNIRS timestamp count mismatch"}
    except IndexError:
        logger.error("fNIRS timestamps array does not have a second column (index 1).")
        return None, {"error": "fNIRS timestamps wrong format"}
    except Exception as e:
        logger.error(f"Could not extract or validate fNIRS timestamps: {e}.")
        return None, {"error": f"fNIRS timestamp extraction error: {e}"}
    
    logger.info(f"Successfully extracted {len(fnirs_ts_ms_orig)} fNIRS timestamps.")
    logger.info(f"Input fnirs_data - Shape: {fnirs_data.shape}, Dtype: {fnirs_data.dtype}, Memory: {fnirs_data.nbytes / (1024*1024):.2f} MB")

    # Attempt to interpolate 0.0 timestamps
    # The 'fnirs_ideal_period_ms' is available in this function's scope.
    initial_ts_len_for_interp = len(fnirs_ts_ms_orig)
    initial_data_len_for_interp = fnirs_data.shape[0]
    
    zero_ts_indices = np.where(fnirs_ts_ms_orig == 0.0)[0]
    
    if len(zero_ts_indices) > 0:
        logger.warning(f"Found {len(zero_ts_indices)} fNIRS frames with 0.0 timestamps. Attempting interpolation.")
        
        for idx in zero_ts_indices:
            # Log data means around the 0.0 timestamp (using finite values)
            if initial_data_len_for_interp > idx:
                current_frame_raw = fnirs_data[idx]
                current_frame_finite = current_frame_raw[np.isfinite(current_frame_raw)]
                mean_current = np.mean(current_frame_finite) if current_frame_finite.size > 0 else np.nan
            else:
                mean_current = np.nan
            log_msg_parts = [f"Frame idx {idx} (original ts=0.0): mean_data={mean_current:.4f}"]

            prev_ts_val = np.nan
            if idx > 0:
                prev_ts_val = fnirs_ts_ms_orig[idx-1]
                if initial_data_len_for_interp > (idx-1):
                    prev_frame_raw = fnirs_data[idx-1]
                    prev_frame_finite = prev_frame_raw[np.isfinite(prev_frame_raw)]
                    mean_prev = np.mean(prev_frame_finite) if prev_frame_finite.size > 0 else np.nan
                else: mean_prev = np.nan
                log_msg_parts.append(f"mean_prev_frame({idx-1})={mean_prev:.4f} (ts={prev_ts_val:.2f})")
            else: log_msg_parts.append("no_prev_frame")

            next_ts_val = np.nan
            if idx < initial_ts_len_for_interp - 1:
                next_ts_val = fnirs_ts_ms_orig[idx+1]
                if initial_data_len_for_interp > (idx+1):
                    next_frame_raw = fnirs_data[idx+1]
                    next_frame_finite = next_frame_raw[np.isfinite(next_frame_raw)]
                    mean_next = np.mean(next_frame_finite) if next_frame_finite.size > 0 else np.nan
                else: mean_next = np.nan
                log_msg_parts.append(f"mean_next_frame({idx+1})={mean_next:.4f} (ts={next_ts_val:.2f})")
            else: log_msg_parts.append("no_next_frame")
            logger.info(f"DEBUG: ZeroTS Info (pre-interpolation): " + ", ".join(log_msg_parts))

            # Attempt interpolation
            interpolated_ts = np.nan
            if idx > 0 and prev_ts_val != 0.0 and not np.isnan(prev_ts_val): # Need a valid previous non-zero timestamp
                # Primary strategy: previous_ts + ideal_period
                candidate_ts = prev_ts_val + fnirs_ideal_period_ms
                
                # Sanity check: ensure candidate_ts is less than the *next actual non-zero* timestamp
                next_actual_ts = np.nan
                for next_i in range(idx + 1, initial_ts_len_for_interp):
                    if fnirs_ts_ms_orig[next_i] != 0.0: # Check against original values before they are changed in this loop
                        next_actual_ts = fnirs_ts_ms_orig[next_i]
                        break
                
                if np.isnan(next_actual_ts) or candidate_ts < next_actual_ts: # If no next actual or candidate is safe
                    interpolated_ts = candidate_ts
                    logger.info(f"Interpolated ts for idx {idx} using prev_ts + ideal_period: {prev_ts_val:.2f} + {fnirs_ideal_period_ms:.2f} = {interpolated_ts:.2f}")
                else:
                    # Secondary strategy: Midpoint, if next_actual_ts is valid and different from prev_ts_val
                    if not np.isnan(next_actual_ts) and prev_ts_val < next_actual_ts :
                        interpolated_ts = (prev_ts_val + next_actual_ts) / 2.0
                        logger.info(f"Interpolated ts for idx {idx} using midpoint (prev_ts={prev_ts_val:.2f}, next_actual_ts={next_actual_ts:.2f}): {interpolated_ts:.2f}")
                    else:
                        logger.warning(f"Could not interpolate ts for idx {idx}: candidate_ts ({candidate_ts:.2f}) >= next_actual_ts ({next_actual_ts:.2f}) or next_actual_ts is invalid or prev_ts_val >= next_actual_ts. Midpoint also not viable.")
            elif idx == 0 and initial_ts_len_for_interp > 1: # First frame has 0.0 ts, try to use next frame
                next_actual_ts_for_first = np.nan
                if fnirs_ts_ms_orig[idx+1] != 0.0:
                    next_actual_ts_for_first = fnirs_ts_ms_orig[idx+1]
                
                if not np.isnan(next_actual_ts_for_first):
                    interpolated_ts = next_actual_ts_for_first - fnirs_ideal_period_ms
                    if interpolated_ts > 0: # Ensure interpolated ts is positive
                         logger.info(f"Interpolated ts for first frame idx {idx} using next_ts - ideal_period: {next_actual_ts_for_first:.2f} - {fnirs_ideal_period_ms:.2f} = {interpolated_ts:.2f}")
                    else:
                        interpolated_ts = np.nan # Revert if it became non-positive
                        logger.warning(f"Could not interpolate ts for first frame idx {idx}: next_ts - ideal_period resulted in non-positive value.")
                else:
                    logger.warning(f"Could not interpolate ts for first frame idx {idx}: next timestamp is also 0.0 or invalid.")
            else: # Handles case where idx is 0 and there's no next frame, or prev_ts_val was 0.0/nan
                logger.warning(f"Could not interpolate ts for idx {idx}: no valid previous non-zero timestamp (or it's the first frame with no valid next for back-interpolation).")

            if not np.isnan(interpolated_ts):
                fnirs_ts_ms_orig[idx] = interpolated_ts # Modify in place
            else:
                logger.warning(f"Timestamp for idx {idx} remains 0.0 as interpolation failed.")
        
        # Log count of remaining 0.0 timestamps if any failed interpolation
        remaining_zero_ts_count = np.sum(fnirs_ts_ms_orig == 0.0)
        if remaining_zero_ts_count > 0:
            logger.warning(f"{remaining_zero_ts_count} timestamps still 0.0 after interpolation attempts.")
        # Frame data (fnirs_data) is NOT filtered/removed here, only timestamps are modified.
    
    # Log shape of fnirs_data *after* this potential timestamp interpolation, before gap processing
    logger.info(f"Post-0.0ts-processing (interpolation attempted): fnirs_data - Shape: {fnirs_data.shape}, Dtype: {fnirs_data.dtype}, Memory: {fnirs_data.nbytes / (1024*1024):.2f} MB. Timestamps count: {len(fnirs_ts_ms_orig)}")

    # --- 1.7. fNIRS Gap Filling and Preprocessing (Adapted from original) ---
    logger.info("Phase 1.7: Processing fNIRS data for gaps...")
    fnirs_ts_ms_processed_list = []
    fnirs_data_processed_list = []
    fnirs_is_actual_list = []

    # For fNIRS-only, the "overlap" is the entire fNIRS stream
    first_orig_fnirs_idx_in_overlap = 0
    last_orig_fnirs_idx_in_overlap = len(fnirs_ts_ms_orig) - 1
    
    if not (first_orig_fnirs_idx_in_overlap <= last_orig_fnirs_idx_in_overlap):
        logger.error("No fNIRS frames to process.")
        return None, {"error": "No fNIRS frames to process"}

    current_actual_ts = fnirs_ts_ms_orig[first_orig_fnirs_idx_in_overlap]
    fnirs_ts_ms_processed_list.append(current_actual_ts)
    fnirs_data_processed_list.append(fnirs_data[first_orig_fnirs_idx_in_overlap])
    fnirs_is_actual_list.append(True)

    for i in range(first_orig_fnirs_idx_in_overlap, last_orig_fnirs_idx_in_overlap):
        ts_of_current_actual_frame_in_loop = fnirs_ts_ms_orig[i]
        ts_of_next_actual_frame_in_loop = fnirs_ts_ms_orig[i+1]
        time_diff = ts_of_next_actual_frame_in_loop - ts_of_current_actual_frame_in_loop
        num_expected_periods = int(round(time_diff / fnirs_ideal_period_ms))
        if num_expected_periods > 1:
            num_gaps_to_fill = num_expected_periods - 1

            last_added_ts_to_processed_list = fnirs_ts_ms_processed_list[-1]
            for k in range(num_gaps_to_fill):
                placeholder_ts = last_added_ts_to_processed_list + fnirs_ideal_period_ms
                fnirs_ts_ms_processed_list.append(placeholder_ts)
                fnirs_data_processed_list.append(np.zeros(fnirs_data[0].shape, dtype=fnirs_data.dtype))
                fnirs_is_actual_list.append(False)
                last_added_ts_to_processed_list = placeholder_ts
        fnirs_ts_ms_processed_list.append(ts_of_next_actual_frame_in_loop)
        fnirs_data_processed_list.append(fnirs_data[i+1])
        fnirs_is_actual_list.append(True)

    fnirs_ts_ms = np.array(fnirs_ts_ms_processed_list)
    fnirs_data_processed = np.stack(fnirs_data_processed_list)
    fnirs_is_actual_frame = np.array(fnirs_is_actual_list)
    num_total_fnirs_frames_processed = len(fnirs_ts_ms)
    num_actual_fnirs_frames_processed = np.sum(fnirs_is_actual_frame)
    
    logger.info(f"fNIRS processing complete (pre-truncation): {num_total_fnirs_frames_processed} total frames ({num_actual_fnirs_frames_processed} actual).")

    if num_actual_fnirs_frames_processed == 0:
        logger.error("No actual fNIRS frames available after gap processing.")
        return None, {"error": "No actual fNIRS after gap processing"}
    if num_total_fnirs_frames_processed < fnirs_frames_per_window:
         logger.error(f"Not enough total fNIRS frames ({num_total_fnirs_frames_processed}) after gap processing.")
         return None, {"error": "Not enough fNIRS post-gap for a window"}

    # --- 1.9: Truncate fNIRS processed data based on ending quality (Adapted) ---
    logger.info("Phase 1.9: Checking fNIRS processed data end quality for truncation...")
    # This logic is largely copied from original create_windows
    original_length_pre_trunc = num_total_fnirs_frames_processed
    if num_total_fnirs_frames_processed >= drift_chunk_size_frames :
        final_valid_idx = num_total_fnirs_frames_processed - 1
        found_good_end_chunk = False
        for current_potential_end_idx in range(num_total_fnirs_frames_processed - 1, drift_chunk_size_frames - 2, -1):
            test_chunk_start_idx = current_potential_end_idx - drift_chunk_size_frames + 1
            if test_chunk_start_idx < 0: break
            chunk_to_test_actual_flags = fnirs_is_actual_frame[test_chunk_start_idx : current_potential_end_idx + 1]
            if len(chunk_to_test_actual_flags) < drift_chunk_size_frames: continue
            density = np.sum(chunk_to_test_actual_flags) / len(chunk_to_test_actual_flags)
            if density >= min_actual_frame_density_for_truncation:
                final_valid_idx = current_potential_end_idx
                found_good_end_chunk = True
                break
        if final_valid_idx < (original_length_pre_trunc - 1):
            if (final_valid_idx + 1) < min_frames_for_late_offset_calc:
                 logger.warning(f"fNIRS data would be too short ({final_valid_idx + 1} frames) after quality truncation. No truncation performed.")
            else:
                logger.info(f"Truncating fNIRS processed data to end at index {final_valid_idx}.")
                fnirs_ts_ms = fnirs_ts_ms[:final_valid_idx + 1]
                fnirs_data_processed = fnirs_data_processed[:final_valid_idx + 1]
                fnirs_is_actual_frame = fnirs_is_actual_frame[:final_valid_idx + 1]
                num_total_fnirs_frames_processed = len(fnirs_ts_ms)
                num_actual_fnirs_frames_processed = np.sum(fnirs_is_actual_frame)
    # (Error checks after truncation)
    if num_actual_fnirs_frames_processed == 0: return None, {"error":"No actual fNIRS post-truncation"}
    if num_total_fnirs_frames_processed < fnirs_frames_per_window: return None, {"error":"Not enough fNIRS post-truncation for a window"}


    # --- Phase 2: fNIRS Timing Characterization (Adapted) ---
    logger.info("Phase 2: Characterizing fNIRS timing and drift...")
    # This logic is largely copied from original create_windows
    avg_early_fnirs_offset = 0.0
    avg_late_fnirs_offset = 0.0
    actual_frame_indices_in_processed = np.where(fnirs_is_actual_frame)[0]
    if num_actual_fnirs_frames_processed > 0:
        num_early_chunk_actual_calc = min(drift_chunk_size_frames, num_actual_fnirs_frames_processed)
        if num_early_chunk_actual_calc > 0:
            early_offsets = []
            # ... (detailed offset calculation as in original) ...
            # Simplified for brevity here, but should be the full logic
            first_actual_ts_for_early_chunk = -1
            actual_frames_count_in_early_chunk = 0
            for i in range(num_early_chunk_actual_calc):
                processed_idx_of_actual_frame = actual_frame_indices_in_processed[i]
                actual_ts = fnirs_ts_ms[processed_idx_of_actual_frame]
                if first_actual_ts_for_early_chunk < 0: first_actual_ts_for_early_chunk = actual_ts
                ideal_ts_in_chunk = first_actual_ts_for_early_chunk + (actual_frames_count_in_early_chunk * fnirs_ideal_period_ms)
                offset = actual_ts - ideal_ts_in_chunk
                early_offsets.append(offset)
                actual_frames_count_in_early_chunk +=1
            if early_offsets: avg_early_fnirs_offset = np.mean(early_offsets)
            logger.info(f"Avg early fNIRS offset (from actuals): {avg_early_fnirs_offset:.3f} ms")
    # Late offset calculation (simplified for brevity)
    if num_total_fnirs_frames_processed >= min_frames_for_late_offset_calc:
        # ... (detailed late offset calculation as in original) ...
        late_offsets_processed = []
        late_chunk_start_idx_processed = num_total_fnirs_frames_processed - drift_chunk_size_frames
        first_ts_for_late_chunk_processed = fnirs_ts_ms[late_chunk_start_idx_processed]
        for i in range(drift_chunk_size_frames):
            current_processed_idx = late_chunk_start_idx_processed + i
            ts_in_processed_chunk = fnirs_ts_ms[current_processed_idx]
            ideal_ts_in_chunk = first_ts_for_late_chunk_processed + (i * fnirs_ideal_period_ms)
            offset = ts_in_processed_chunk - ideal_ts_in_chunk
            late_offsets_processed.append(offset)
        if late_offsets_processed: avg_late_fnirs_offset = np.mean(late_offsets_processed)
        logger.info(f"Avg late fNIRS offset (from processed): {avg_late_fnirs_offset:.3f} ms")

    else:
        avg_late_fnirs_offset = avg_early_fnirs_offset # Fallback

    if len(actual_frame_indices_in_processed) == 0: return None, {"error": "No actual fNIRS for timing projection"}
    first_actual_fnirs_ts_in_processed = fnirs_ts_ms[actual_frame_indices_in_processed[0]]
    projected_first_window_start_ts = first_actual_fnirs_ts_in_processed - avg_early_fnirs_offset
    logger.info(f"Projected start timestamp for the first fNIRS window: {projected_first_window_start_ts:.3f} ms")


    # --- Phase 3: Iterate to Create fNIRS Windows ---
    logger.info("Phase 3: Iterating to create fNIRS windows...")
    windows_list: List[Dict[str, Union[np.ndarray, Dict]]] = []
    current_fnirs_processed_idx = 0
    window_count = 0

    while True:
        fnirs_chunk_end_processed_idx = current_fnirs_processed_idx + fnirs_frames_per_window - 1
        if fnirs_chunk_end_processed_idx >= num_total_fnirs_frames_processed:
            logger.info(f"Stopping: Not enough processed fNIRS frames for window {window_count}.")
            break

        fnirs_w_np = fnirs_data_processed[current_fnirs_processed_idx : current_fnirs_processed_idx + fnirs_frames_per_window]
        fnirs_ts_window_ms = fnirs_ts_ms[current_fnirs_processed_idx : current_fnirs_processed_idx + fnirs_frames_per_window]
        is_actual_fnirs = fnirs_is_actual_frame[current_fnirs_processed_idx : current_fnirs_processed_idx + fnirs_frames_per_window] # This is a slice now
        
        window_ideal_fnirs_event_ts = projected_first_window_start_ts + (window_count * fnirs_ideal_period_ms)
        actual_this_fnirs_event_ts = fnirs_ts_window_ms[0] # First timestamp in the fNIRS window

        metadata = {
            'window_idx': window_count,
            'fnirs_frame_indices_processed': (current_fnirs_processed_idx, fnirs_chunk_end_processed_idx),
            'fnirs_timestamps_ms': fnirs_ts_window_ms,
            'fnirs_frame_is_actual': is_actual_fnirs.all(), # Mark true if all frames in window are actual
            'ideal_fnirs_event_ts_projected': window_ideal_fnirs_event_ts,
            'actual_fnirs_event_ts': actual_this_fnirs_event_ts,
            'notes': "fNIRS-only window."
        }
        windows_list.append({'fnirs': fnirs_w_np, 'metadata': metadata})
        current_fnirs_processed_idx += fnirs_frames_per_window
        window_count += 1
        
    if not windows_list: logger.warning("No fNIRS windows were created.")
    else: logger.info(f"Successfully created {len(windows_list)} fNIRS windows.")

    # --- Phase 4: Prepare for NumPy Array Output ---
    logger.info("Phase 4: Transforming fNIRS windows_list to NumPy array outputs.")
    raw_fnirs_ts_ms = fnirs_ts_ms_orig # Original, full fnirs timestamps

    overall_min_ts = raw_fnirs_ts_ms[0]
    overall_max_ts = raw_fnirs_ts_ms[-1] + fnirs_ideal_period_ms
    
    if overall_max_ts <= overall_min_ts: total_windows = 0
    else: total_windows = int(np.ceil((overall_max_ts - overall_min_ts) / fnirs_ideal_period_ms))

    if total_windows == 0: logger.warning("Master timeline results in 0 total_windows.")
    main_clock_timestamps_array = overall_min_ts + np.arange(total_windows) * fnirs_ideal_period_ms

    num_fnirs_chans = fnirs_data.shape[1]
    fnirs_samples_per_raw_frame = fnirs_data.shape[2] # Should be 1

    fnirs_windows_data_arr = np.zeros((total_windows, num_fnirs_chans, fnirs_samples_per_raw_frame * fnirs_frames_per_window), dtype=fnirs_data.dtype)
    fnirs_validity_mask_arr = np.zeros(total_windows, dtype=bool)
    real_fnirs_timestamps_arr = np.full(total_windows, np.nan, dtype=float)

    # Empty EEG arrays
    num_eeg_chans_dummy = 1 # Placeholder
    eeg_samples_per_output_window_dummy = 1 # Placeholder
    eeg_windows_data_arr = np.zeros((total_windows, num_eeg_chans_dummy, eeg_samples_per_output_window_dummy), dtype=np.float32) # Placeholder
    eeg_validity_mask_arr = np.zeros(total_windows, dtype=bool)
    real_eeg_timestamps_arr = np.full(total_windows, np.nan, dtype=float)

    if windows_list:
        for original_window in windows_list:
            map_key_ts = original_window['metadata'].get('ideal_fnirs_event_ts_projected', original_window['metadata']['actual_fnirs_event_ts'])
            if map_key_ts is None: continue

            if total_windows > 0:
                diffs = np.abs(main_clock_timestamps_array - map_key_ts)
                slot_idx = np.argmin(diffs)
                if diffs[slot_idx] > (fnirs_ideal_period_ms / 1.9): continue
            else: continue
            
            if original_window['metadata']['fnirs_frame_is_actual']:
                fnirs_from_list = original_window['fnirs']
                expected_shape = (fnirs_frames_per_window, num_fnirs_chans, fnirs_samples_per_raw_frame)
                if fnirs_from_list.shape == expected_shape:
                    # Reshape to (chans, samples_in_window)
                    fnirs_windows_data_arr[slot_idx] = fnirs_from_list.transpose(1,0,2).reshape(num_fnirs_chans, fnirs_samples_per_raw_frame * fnirs_frames_per_window)
                    fnirs_validity_mask_arr[slot_idx] = True
                    real_fnirs_timestamps_arr[slot_idx] = original_window['metadata']['actual_fnirs_event_ts']
                else:
                    logger.warning(f"Actual fNIRS data in original_window has unexpected shape {fnirs_from_list.shape}. Expected {expected_shape}.")
    
    def get_span_indices(mask_arr): # Redefined locally
        valid_indices = np.where(mask_arr)[0]
        if len(valid_indices) > 0: return valid_indices[0], valid_indices[-1] + 1
        return -1, -1

    fnirs_start_idx, fnirs_end_idx = get_span_indices(fnirs_validity_mask_arr)
    if fnirs_start_idx != -1: fnirs_start_idx = int(fnirs_start_idx)
    if fnirs_end_idx != -1: fnirs_end_idx = int(fnirs_end_idx)
    eeg_start_idx, eeg_end_idx = -1, -1 # No EEG data

    # For fNIRS-only, "both" means fNIRS
    both_valid_mask = fnirs_validity_mask_arr
    both_start_idx, both_end_idx = get_span_indices(both_valid_mask)
    if both_start_idx != -1: both_start_idx = int(both_start_idx)
    if both_end_idx != -1: both_end_idx = int(both_end_idx)

    percent_fnirs_trimmed = 0.0 # Simplified
    percent_fnirs_missing = 0.0 # Simplified
    # (More detailed calculation can be added like in original create_windows)

    summary_metadata = {
        'percent_eeg_trimmed': 100.0, # No EEG
        'percent_fnirs_trimmed': percent_fnirs_trimmed,
        'percent_fnirs_missing_in_span': percent_fnirs_missing,
        'total_master_windows_in_input_arrays': total_windows,
        'eeg_master_span_in_input_arrays': (eeg_start_idx, eeg_end_idx),
        'fnirs_master_span_in_input_arrays': (fnirs_start_idx, fnirs_end_idx),
        'both_master_span_in_input_arrays': (both_start_idx, both_end_idx) # 'both' is fNIRS here
    }
    
    # Pass through retained_fnirs_indices if they exist in metadata
    if metadata and 'retained_fnirs_indices' in metadata:
        summary_metadata['retained_fnirs_indices'] = metadata['retained_fnirs_indices']
    logger.info(f"NumPy array creation complete for fNIRS-only. Total master windows: {total_windows}")
    logger.info(f"fNIRS: {np.sum(fnirs_validity_mask_arr)} valid windows. Span: {fnirs_start_idx}-{fnirs_end_idx}.")

    # --- Phase 5: Instantiate WindowDataset and Return ---
    logger.info("Phase 5: Instantiating WindowDataset for fNIRS-only.")
    dataset_instance: Optional[WindowDataset] = None
    if total_windows == 0:
        logger.warning("Total master windows is 0 for fNIRS-only.")
    elif not (isinstance(both_start_idx, int) and isinstance(both_end_idx, int) and \
            both_start_idx != -1 and both_end_idx != -1 and both_start_idx < both_end_idx):
        logger.warning(f"Cannot create WindowDataset: No valid span for fNIRS data. "
                       f"both_start_idx (fnirs_start_idx): {both_start_idx}, both_end_idx (fnirs_end_idx): {both_end_idx}")
    else:
        try:
            dataset_instance = WindowDataset(
                eeg_windows_data_arr=eeg_windows_data_arr, # Empty/zero
                fnirs_windows_data_arr=fnirs_windows_data_arr,
                eeg_validity_mask_arr=eeg_validity_mask_arr, # All False
                fnirs_validity_mask_arr=fnirs_validity_mask_arr,
                real_eeg_timestamps_arr=real_eeg_timestamps_arr, # All NaN
                real_fnirs_timestamps_arr=real_fnirs_timestamps_arr,
                main_clock_timestamps_array=main_clock_timestamps_array,
                eeg_start_idx=eeg_start_idx, # -1
                fnirs_start_idx=fnirs_start_idx,
                eeg_end_idx=eeg_end_idx, # -1
                fnirs_end_idx=fnirs_end_idx,
                both_start_idx=both_start_idx, # This is fnirs_start_idx
                both_end_idx=both_end_idx,   # This is fnirs_end_idx
                return_torch_tensors=return_torch_tensors
            )
            logger.info(f"Successfully created WindowDataset with {len(dataset_instance)} fNIRS-only windows.")
        except ValueError as ve:
            logger.error(f"Error instantiating WindowDataset for fNIRS-only: {ve}")
            dataset_instance = None
        except Exception as e:
            logger.error(f"Unexpected error during WindowDataset instantiation for fNIRS-only: {e}", exc_info=True)
            dataset_instance = None
            
    logger.info(f"Finalizing create_windows_fnirs_only. summary_metadata keys: {list(summary_metadata.keys())}")
    return dataset_instance, summary_metadata

