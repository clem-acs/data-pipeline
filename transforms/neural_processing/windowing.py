import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional, Union

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
    return_torch_tensors: bool = True
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

    # --- 4. Instantiate WindowDataset ---
    if not windows_list:
        logger.warning("No windows were generated. Returning None for WindowDataset.")
        return None 
    try:
        from .window_dataset import WindowDataset 
        dataset = WindowDataset(windows_list=windows_list, return_torch_tensors=return_torch_tensors)
        logger.info(f"Successfully instantiated WindowDataset with {len(dataset)} windows.")
        return dataset
    except ImportError: 
        logger.error("Failed to import WindowDataset. Ensure it's correctly placed and project is structured as a package for relative imports.", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Failed to instantiate WindowDataset: {e}", exc_info=True)
        return None
