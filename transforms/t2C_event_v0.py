"""
T2E Events Transform

Extracts and organizes event data from curated H5 files into a structured format
using xarray and zarr for efficient storage and access.

IMPORTANT: In our H5 files, timestamps are stored as [server_timestamp, client_timestamp].
That is, timestamps[i][0] is the server timestamp and timestamps[i][1] is the client timestamp.
"""

import os
import json
import logging
import numpy as np
import h5py
import time
import xarray as xr
import zarr
from typing import Dict, Any, List, Optional
from collections import defaultdict

# Import base transform
from base_transform import BaseTransform, Session

# Constants for default values
DEFAULT_AUDIO_MODE = "text_only"
DEFAULT_INPUT_MODALITY = "text"

# Module logger - only used before class initialization
logger = logging.getLogger(__name__)

def _convert_to_string_dtype(values):
    """Convert values to proper string dtype for Zarr storage.
    
    Compatible with all zarr versions by using NumPy's standard string dtype.
    """
    # Replace None with empty strings
    cleaned_values = ['' if v is None else v for v in values]
    # Use NumPy's Unicode string dtype - compatible with all zarr versions
    return np.array(cleaned_values, dtype='U')

class EventTransform(BaseTransform):
    """Transform for extracting and organizing event data from H5 files into zarr format."""
    
    SOURCE_PREFIX = "curated-h5/"
    DEST_PREFIX = "processed/event/"
    
    def __init__(self, **kwargs):
        """
        Initialize the events transform.

        Args:
            **kwargs: Additional arguments for BaseTransform
        """
        # Set default transform info if not provided
        transform_id = kwargs.pop('transform_id', 't2C_event_v0')
        script_id = kwargs.pop('script_id', '2C')
        script_name = kwargs.pop('script_name', 'event')
        script_version = kwargs.pop('script_version', 'v0')

        # Call parent constructor
        super().__init__(
            transform_id=transform_id,
            script_id=script_id,
            script_name=script_name,
            script_version=script_version,
            **kwargs
        )
        
        self.logger.info(f"Event extraction transform initialized")
    
    def process_session(self, session: Session) -> Dict:
        """Process a single session, extracting event data into a structured format.
        
        Args:
            session: Session object
            
        Returns:
            Dict: Dictionary with processing results
        """
        session_id = session.session_id
        self.logger.info(f"Processing events for session {session_id}")
        
        try:
            # Look for H5 file in the source prefix
            h5_key = f"{self.source_prefix}{session_id}.h5"
            
            # Check if the file exists
            try:
                self.s3.head_object(Bucket=self.s3_bucket, Key=h5_key)
                self.logger.info(f"Found H5 file: {h5_key}")
            except Exception as e:
                self.logger.error(f"No H5 file found for session {session_id}: {e}")
                return {
                    "status": "failed",
                    "error_details": f"No H5 file found for session {session_id}",
                    "metadata": {"session_id": session_id},
                    "files_to_copy": [],
                    "files_to_upload": [],
                    "zarr_stores": []
                }
            
            # Download the H5 file
            local_source = session.download_file(h5_key)
            
            # Process the file
            with h5py.File(local_source, 'r') as source_h5:
                # Extract and process events
                self.logger.info(f"Extracting events for {session_id}")
                events = self._extract_raw_events(source_h5)
                processed_data = self._process_event_data(events, session_id, source_h5)
                
                if not processed_data:
                    self.logger.error(f"Failed to process events for {session_id}")
                    return {
                        "status": "failed",
                        "error_details": f"Failed to process events for {session_id}",
                        "metadata": {"session_id": session_id},
                        "files_to_copy": [],
                        "files_to_upload": [],
                        "zarr_stores": []
                    }
            
            # Create xarray dataset from processed data
            dataset = self._create_xarray_dataset(processed_data)
            
            # Check for empty dataset
            if not dataset.data_vars:
                self.logger.warning(f"No data to save for session {session_id}, creating empty dataset")
                # Create minimal dataset with a single variable and dimension for chunking safety
                dataset = xr.Dataset(
                    data_vars={
                        'empty_flag': (['empty_dim'], np.array([True]))
                    },
                    coords={
                        'empty_dim': [0]
                    },
                    attrs={
                        'session_id': session_id,
                        'transform': 'event',
                        'version': '0.1',
                        'created_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                        'empty': True
                    }
                )
            else:
                self.logger.info(f"Dataset contains {len(dataset.data_vars)} data variables")
            
            # Define the destination zarr key
            zarr_key = f"{self.destination_prefix}{session_id}_events.zarr"
            
            # Create explicit chunks based on dimensions (one chunk per dimension)
            chunks = {}
            for dim_name, dim_size in dataset.dims.items():
                if dim_size > 0:
                    chunks[dim_name] = dim_size
            
            self.logger.info(f"Saving hierarchical dataset with {len(dataset.data_vars)} variables to {zarr_key}")
            self.logger.debug(f"Dataset contains groups: elements, tasks, segments")
            self.logger.debug(f"Dataset coordinates: {list(dataset.coords.keys())}")
            
            # Directly save to S3 using BaseTransform's method with explicit chunks
            self.save_dataset_to_s3_zarr(dataset, zarr_key, chunks=chunks)
            
            # Create metadata for DynamoDB
            metadata = {
                "session_id": session_id,
                "processed_at": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                "storage_format": "zarr_xarray",
                "task_count": len(processed_data['tasks']),
                "element_count": len(processed_data['elements']),
                "segment_count": sum(len(segments) for segments in processed_data['segments'].values()),
                "segment_types": list(processed_data['segments'].keys())
            }
            
            self.logger.info(f"Successfully processed events for {session_id}")
            return {
                "status": "success",
                "metadata": metadata,
                "files_to_copy": [],
                "files_to_upload": [],
                "zarr_stores": [zarr_key]
            }
            
        except Exception as e:
            self.logger.error(f"Error processing events for {session_id}: {e}", exc_info=True)
            return {
                "status": "failed",
                "error_details": str(e),
                "metadata": {"session_id": session_id},
                "files_to_copy": [],
                "files_to_upload": [],
                "zarr_stores": []
            }
    
    def _create_xarray_dataset(self, processed_data):
        """Convert processed event data into a hierarchical xarray Dataset.
        
        Args:
            processed_data: Dictionary of processed tasks, elements, and segments
            
        Returns:
            xarray Dataset with hierarchical structure using sub-groups
        """
        # Extract component data
        tasks = processed_data['tasks']
        elements = processed_data['elements']
        segments = processed_data['segments']
        session_id = processed_data['session_id']
        
        # Initialize dataset groups
        elements_group = {}
        tasks_group = {}
        segments_group = {}
        
        # ------------ TASK DATA ------------
        task_ids = list(tasks.keys())
        
        if task_ids:
            # Create arrays for each task attribute
            task_arrays = defaultdict(list)
            
            # Define which task attributes to store
            task_attributes = [
                'task_type', 'start_time', 'end_time', 'duration', 
                'completion_status', 'session_fraction_start', 'session_fraction_end',
                'count', 'allow_repeats', 'with_interruptions', 'input_modality', 
                'audio_mode', 'skipped', 'skip_time', 'element_count'
            ]
            
            # Fill arrays with task data
            for task_id in task_ids:
                task = tasks[task_id]
                for attr in task_attributes:
                    # Use empty/default values if attribute doesn't exist
                    task_arrays[attr].append(task.get(attr, None))
            
            # Store task data
            for attr, values in task_arrays.items():
                if all(isinstance(v, str) or v is None for v in values):
                    # Use string dtype for string values
                    values = _convert_to_string_dtype(values)
                else:
                    # For numeric arrays, replace None with appropriate default
                    if any(v is None for v in values):
                        non_none_values = [v for v in values if v is not None]
                        if non_none_values:
                            if all(isinstance(v, (int, np.integer)) for v in non_none_values):
                                cleaned_values = [0 if v is None else v for v in values]
                                values = np.array(cleaned_values, dtype=np.int32)
                            elif all(isinstance(v, (float, np.floating)) for v in non_none_values):
                                cleaned_values = [0.0 if v is None else v for v in values]
                                values = np.array(cleaned_values, dtype=np.float64)
                            elif all(isinstance(v, bool) for v in non_none_values):
                                cleaned_values = [False if v is None else v for v in values]
                                values = np.array(cleaned_values, dtype=bool)
                            else:
                                # Mixed or unknown type, use strings
                                values = _convert_to_string_dtype(['' if v is None else str(v) for v in values])
                        else:
                            # All None values, use empty strings
                            values = _convert_to_string_dtype([''] * len(values))
                    else:
                        # No None values, convert normally
                        values = np.array(values)
                
                # Store in tasks group with proper attribute name (no prefix)
                tasks_group[attr] = ('task_id', values)
        
        # ------------ ELEMENT DATA ------------
        element_ids = list(elements.keys())
        
        if element_ids:
            # Create arrays for each element attribute
            element_arrays = defaultdict(list)
            
            # Define which element attributes to store
            element_attributes = [
                'element_type', 'start_time', 'end_time', 'duration',
                'title', 'is_instruction', 'task_id', 'sequence_idx', 'max_count',
                'session_fraction', 'session_relative_time', 'input_modality', 
                'audio_mode', 'with_interruptions', 'response_time_seconds'
            ]
            
            # Fill arrays with element data
            for element_id in element_ids:
                element = elements[element_id]
                
                # Add task_type from the associated task
                task_id = element.get('task_id', '')
                if task_id and task_id in tasks:
                    element['task_type'] = tasks[task_id].get('task_type', '')
                else:
                    element['task_type'] = ''
                
                for attr in element_attributes:
                    element_arrays[attr].append(element.get(attr, None))
                
                # Add task_type if not already in attributes
                if 'task_type' not in element_attributes:
                    element_arrays['task_type'].append(element.get('task_type', ''))
            
            # Store element data
            for attr, values in element_arrays.items():
                if all(isinstance(v, str) or v is None for v in values):
                    # Use string dtype for string values
                    values = _convert_to_string_dtype(values)
                else:
                    # For numeric arrays, replace None with appropriate default
                    if any(v is None for v in values):
                        non_none_values = [v for v in values if v is not None]
                        if non_none_values:
                            if all(isinstance(v, (int, np.integer)) for v in non_none_values):
                                cleaned_values = [0 if v is None else v for v in values]
                                values = np.array(cleaned_values, dtype=np.int32)
                            elif all(isinstance(v, (float, np.floating)) for v in non_none_values):
                                cleaned_values = [0.0 if v is None else v for v in values]
                                values = np.array(cleaned_values, dtype=np.float64)
                            elif all(isinstance(v, bool) for v in non_none_values):
                                cleaned_values = [False if v is None else v for v in values]
                                values = np.array(cleaned_values, dtype=bool)
                            else:
                                # Mixed or unknown type, use strings
                                values = _convert_to_string_dtype(['' if v is None else str(v) for v in values])
                        else:
                            # All None values, use empty strings
                            values = _convert_to_string_dtype([''] * len(values))
                    else:
                        # No None values, convert normally
                        values = np.array(values)
                
                # Store in elements group with proper attribute name (no prefix)
                elements_group[attr] = ('element_id', values)
        
        # ------------ SEGMENT DATA ------------
        segment_types = list(segments.keys())
        
        if segment_types:
            all_segment_ids = []
            segment_arrays = defaultdict(list)
            segment_type_indices = []
            
            # Define which segment attributes to store
            segment_attributes = [
                'segment_type', 'start_time', 'end_time', 'duration',
                'containing_element_id', 'element_relative_start',
                'start_event_id', 'end_event_id'
            ]
            
            # First pass - collect all unique segment ids and data
            for segment_type, segment_list in segments.items():
                for segment in segment_list:
                    segment_id = segment['segment_id']
                    all_segment_ids.append(segment_id)
                    segment_type_indices.append(segment_type)
                    
                    for attr in segment_attributes:
                        segment_arrays[attr].append(segment.get(attr, None))
            
            # Store segment data
            for attr, values in segment_arrays.items():
                if all(isinstance(v, str) or v is None for v in values):
                    # Use string dtype for string values
                    values = _convert_to_string_dtype(values)
                else:
                    # For numeric arrays, replace None with appropriate default
                    if any(v is None for v in values):
                        non_none_values = [v for v in values if v is not None]
                        if non_none_values:
                            if all(isinstance(v, (int, np.integer)) for v in non_none_values):
                                cleaned_values = [0 if v is None else v for v in values]
                                values = np.array(cleaned_values, dtype=np.int32)
                            elif all(isinstance(v, (float, np.floating)) for v in non_none_values):
                                cleaned_values = [0.0 if v is None else v for v in values]
                                values = np.array(cleaned_values, dtype=np.float64)
                            elif all(isinstance(v, bool) for v in non_none_values):
                                cleaned_values = [False if v is None else v for v in values]
                                values = np.array(cleaned_values, dtype=bool)
                            else:
                                # Mixed or unknown type, use strings
                                values = _convert_to_string_dtype(['' if v is None else str(v) for v in values])
                        else:
                            # All None values, use empty strings
                            values = _convert_to_string_dtype([''] * len(values))
                    else:
                        # No None values, convert normally
                        values = np.array(values)
                
                # Store in segments group with proper attribute name (no prefix)
                segments_group[attr] = ('segment_id', values)
            
            # Convert segment_type_indices to fixed-length string dtype
            segment_type_indices_array = _convert_to_string_dtype(segment_type_indices)
            segments_group['type_index'] = ('segment_id', segment_type_indices_array)
        
        # Create data variables and coordinates for the entire dataset
        data_vars = {}
        coords = {}
        
        # Create hierarchical structure using netCDF groups
        if task_ids:
            # Store task data in tasks group
            for attr, (dim, values) in tasks_group.items():
                data_vars[f"tasks/{attr}"] = (["task_id"], values)
            coords["task_id"] = task_ids
        
        if element_ids:
            # Store element data in elements group
            for attr, (dim, values) in elements_group.items():
                data_vars[f"elements/{attr}"] = (["element_id"], values)
            coords["element_id"] = element_ids
        
        if all_segment_ids:
            # Store segment data in segments group
            for attr, (dim, values) in segments_group.items():
                data_vars[f"segments/{attr}"] = (["segment_id"], values)
            coords["segment_id"] = all_segment_ids
        
        # Create dataset with hierarchical structure
        ds = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs={
                'session_id': session_id,
                'transform': 'event',
                'version': '0.1',
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            }
        )
        
        # Create explicit chunks based on dimensions (one chunk per dimension)
        chunks = {}
        for dim_name, dim_size in ds.dims.items():
            if dim_size > 0:
                chunks[dim_name] = dim_size
        
        # Apply chunking
        if chunks:
            ds = ds.chunk(chunks)
        
        # Add explicit chunk sizes for coordinates to avoid None values
        for name in ds.coords:
            if ds[name].encoding.get("chunks") is None:
                ds[name].encoding["chunks"] = ds[name].shape
        
        return ds
    
    def _extract_raw_events(self, h5_file):
        """Extract raw events from the H5 file.

        Args:
            h5_file: H5 file object

        Returns:
            dict: Dictionary of event data by type
        """
        events = {}

        # Check if events group exists
        if 'events' not in h5_file:
            self.logger.warning("No events group in H5 file")
            return events

        # Extract all event types
        for event_type in h5_file['events']:
            events[event_type] = {
                'data': [],
                'event_ids': [],
                'timestamps': []
            }

            # Extract event data
            if 'data' in h5_file[f'events/{event_type}']:
                data = h5_file[f'events/{event_type}/data'][:]
                events[event_type]['data'] = [json.loads(d) for d in data]

            # Extract event IDs
            if 'event_ids' in h5_file[f'events/{event_type}']:
                event_ids = h5_file[f'events/{event_type}/event_ids'][:]
                events[event_type]['event_ids'] = [eid.decode('utf-8') if isinstance(eid, bytes) else eid for eid in event_ids]

            # Extract timestamps
            if 'timestamps' in h5_file[f'events/{event_type}']:
                events[event_type]['timestamps'] = h5_file[f'events/{event_type}/timestamps'][:]

                # DEBUG: Print first few timestamps to understand their structure
                if len(events[event_type]['timestamps']) > 0:
                    print(f"DEBUG-TS-EXTRACT: Event type {event_type} has timestamps shape {events[event_type]['timestamps'].shape}")
                    print(f"DEBUG-TS-EXTRACT: First timestamp array: {events[event_type]['timestamps'][0]}")
                    print(f"DEBUG-TS-EXTRACT: Using index 0 as client timestamp: {events[event_type]['timestamps'][0][0]}")
                    print(f"DEBUG-TS-EXTRACT: Using index 1 as server timestamp: {events[event_type]['timestamps'][0][1]}")

        return events
    
    def _process_segment_type(self, events, start_event_type, end_event_type, segment_type, segments):
        """Process segments of a specific type.
        
        Args:
            events: Dictionary of events
            start_event_type: Type of start event (e.g., 'recording_start')
            end_event_type: Type of end event (e.g., 'recording_stop')
            segment_type: Type of segment (e.g., 'recording')
            segments: Dictionary to store segments
        """
        # Get event pairs
        pairs = self._pair_events(events, start_event_type, end_event_type)
        
        # Process each pair
        for i, (start_id, (start_event, end_event)) in enumerate(pairs.items()):
            segment_id = start_id
            start_time = start_event['client_timestamp']
            end_time = end_event['client_timestamp']

            # DEBUG: Print segment timestamps to confirm what's being used
            print(f"DEBUG-TS-SEGMENT: Creating segment {segment_type}/{segment_id}")
            print(f"DEBUG-TS-SEGMENT: start_event client timestamp: {start_event['client_timestamp']}")
            print(f"DEBUG-TS-SEGMENT: start_event server timestamp: {start_event['server_timestamp']}")
            print(f"DEBUG-TS-SEGMENT: Using as segment start_time: {start_time}")

            segments[segment_type].append({
                'segment_id': segment_id,
                'segment_type': segment_type,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'start_event_id': start_event['event_ids'][0],
                'end_event_id': end_event['event_ids'][0],
                'containing_element_id': '',
                'element_relative_start': 0.0
            })

    def _extract_timestamp(self, timestamps, idx, timestamp_type):
        """Extract timestamp safely with proper index.

        Args:
            timestamps: Timestamps array
            idx: Index into the array
            timestamp_type: 'client' or 'server' to select correct column

        Returns:
            float: Extracted timestamp or 0.0 if not available
        """
        if idx >= len(timestamps):
            return 0.0

        col_idx = 1 if timestamp_type == 'client' else 0  # [server_ts, client_ts]
        return float(timestamps[idx][col_idx])

    def _is_empty_or_none(self, value):
        """Check if a value is None, empty string, or empty bytes.

        Args:
            value: Value to check

        Returns:
            bool: True if value is None or empty, False otherwise
        """
        if value is None:
            return True
        if isinstance(value, (str, bytes)) and len(value) == 0:
            return True
        return False

    def _is_segment_contained_in_element(self, segment_time, element, segment_id=None, element_id=None):
        """Check if a segment timestamp is contained within an element's timespan.

        Args:
            segment_time: Segment start time to check
            element: Element dictionary with start_time and end_time
            segment_id: Optional segment ID for debug output
            element_id: Optional element ID for debug output

        Returns:
            bool: True if segment is contained in element, False otherwise
        """
        # Debug output if IDs are provided
        if segment_id and element_id:
            print(f"DEBUG-TS-CONTAIN: Containment check for segment={segment_id} in element={element_id}")
            # print(f"DEBUG-TS-CONTAIN: Segment start_time={segment_time}")
            # print(f"DEBUG-TS-CONTAIN: Element start_time={element['start_time']}, end_time={element['end_time']}")
            # print(f"DEBUG-TS-CONTAIN: Condition 1: element.start <= segment.start = {element['start_time'] <= segment_time}")
            # print(f"DEBUG-TS-CONTAIN: Condition 2: element.end >= segment.start = {element['end_time'] >= segment_time}")
            # print(f"DEBUG-TS-CONTAIN: Combined condition = {(element['start_time'] <= segment_time and element['end_time'] >= segment_time)}")

        # The actual containment check
        return (element['start_time'] <= segment_time and
                element['end_time'] >= segment_time)

    def _pair_events(self, events, start_type, end_type):
        """Pair start/end events.

        Args:
            events: Dictionary of events
            start_type: Type of start event
            end_type: Type of end event

        Returns:
            dict: Dictionary of paired events by ID
        """
        pairs = {}

        # Check if both event types exist
        if start_type not in events or end_type not in events:
            return pairs

        start_events = events[start_type]
        end_events = events[end_type]

        # Special case for element events - match by element_id
        if start_type == 'element_sent' and end_type == 'element_replied':
            # Create dictionaries to map element_id -> event data
            start_elements_by_id = {}
            end_elements_by_id = {}

            # Process all sent events, organizing by element_id
            for i in range(len(start_events['event_ids'])):
                if i < len(start_events['data']) and 'element_id' in start_events['data'][i]:
                    element_id = start_events['data'][i]['element_id']
                    start_id = start_events['event_ids'][i]

                    # Extract timestamps - NOTE: In H5 files, timestamps are [server_timestamp, client_timestamp]
                    client_ts = float(start_events['timestamps'][i][1]) if i < len(start_events['timestamps']) else 0.0
                    server_ts = float(start_events['timestamps'][i][0]) if i < len(start_events['timestamps']) else 0.0

                    # Store with all necessary data
                    start_elements_by_id[element_id] = {
                        'data': start_events['data'][i] if i < len(start_events['data']) else {},
                        'event_ids': [start_id],
                        'client_timestamp': client_ts,
                        'server_timestamp': server_ts
                    }

            # Process all replied events, organizing by element_id
            for i in range(len(end_events['event_ids'])):
                if i < len(end_events['data']) and 'element_id' in end_events['data'][i]:
                    element_id = end_events['data'][i]['element_id']
                    end_id = end_events['event_ids'][i]

                    # Extract timestamps - NOTE: In H5 files, timestamps are [server_timestamp, client_timestamp]
                    client_ts = float(end_events['timestamps'][i][1]) if i < len(end_events['timestamps']) else 0.0
                    server_ts = float(end_events['timestamps'][i][0]) if i < len(end_events['timestamps']) else 0.0

                    # Store with all necessary data
                    end_elements_by_id[element_id] = {
                        'data': end_events['data'][i] if i < len(end_events['data']) else {},
                        'event_ids': [end_id],
                        'client_timestamp': client_ts,
                        'server_timestamp': server_ts
                    }

            # Now match elements by element_id (instead of by index)
            for element_id, start_data in start_elements_by_id.items():
                if element_id in end_elements_by_id:
                    end_data = end_elements_by_id[element_id]
                    start_id = start_data['event_ids'][0]

                    # DEBUG: Print raw timestamps for this pair
                    # print(f"DEBUG-TS-PAIR: Element pair {start_type}/{end_type} - {start_id}/{end_data['event_ids'][0]} for element_id={element_id}")
                    # print(f"DEBUG-TS-PAIR: Start timestamps: client={start_data['client_timestamp']}, server={start_data['server_timestamp']}")
                    # print(f"DEBUG-TS-PAIR: End timestamps: client={end_data['client_timestamp']}, server={end_data['server_timestamp']}")

                    # Create the pair with proper timestamps
                    pairs[start_id] = (start_data, end_data)
        else:
            # Original logic for non-element events
            # Simple matching by index if counts match
            if len(start_events['event_ids']) == len(end_events['event_ids']):
                for i in range(len(start_events['event_ids'])):
                    start_id = start_events['event_ids'][i]
                    end_id = end_events['event_ids'][i]

                    # Extract timestamps - NOTE: In H5 files, timestamps are [server_timestamp, client_timestamp]
                    client_ts_start = self._extract_timestamp(start_events['timestamps'], i, 'client')
                    server_ts_start = self._extract_timestamp(start_events['timestamps'], i, 'server')
                    client_ts_end = self._extract_timestamp(end_events['timestamps'], i, 'client')
                    server_ts_end = self._extract_timestamp(end_events['timestamps'], i, 'server')

                    # DEBUG: Print raw timestamps for this pair
                    # print(f"DEBUG-TS-PAIR: Event pair {start_type}/{end_type} - {start_id}/{end_id}")
                    # print(f"DEBUG-TS-PAIR: Start timestamps: client={client_ts_start}, server={server_ts_start}")
                    # print(f"DEBUG-TS-PAIR: End timestamps: client={client_ts_end}, server={server_ts_end}")

                    start_data = {
                        'data': start_events['data'][i] if i < len(start_events['data']) else {},
                        'event_ids': [start_id],
                        'client_timestamp': client_ts_start,
                        'server_timestamp': server_ts_start
                    }

                    end_data = {
                        'data': end_events['data'][i] if i < len(end_events['data']) else {},
                        'event_ids': [end_id],
                        'client_timestamp': client_ts_end,
                        'server_timestamp': server_ts_end
                    }

                    pairs[start_id] = (start_data, end_data)
            else:
                # More complex matching by timestamps
                self.logger.warning(f"Mismatched counts for {start_type}/{end_type}, using timestamp matching")

                # Sort by timestamp - NOTE: In H5 files, timestamps are [server_timestamp, client_timestamp]
                start_items = sorted(list(zip(
                    start_events['event_ids'],
                    start_events['timestamps'][:, 1] if len(start_events['timestamps']) > 0 else [],  # Use client timestamp
                    range(len(start_events['event_ids']))
                )), key=lambda x: x[1])

                end_items = sorted(list(zip(
                    end_events['event_ids'],
                    end_events['timestamps'][:, 1] if len(end_events['timestamps']) > 0 else [],  # Use client timestamp
                    range(len(end_events['event_ids']))
                )), key=lambda x: x[1])

                # Match each start with the next end
                for i, (start_id, start_time, start_idx) in enumerate(start_items):
                    if i < len(end_items):
                        end_id, end_time, end_idx = end_items[i]

                        if end_time >= start_time:  # Ensure end is after start
                            start_data = {
                                'data': start_events['data'][start_idx] if start_idx < len(start_events['data']) else {},
                                'event_ids': [start_id],
                                'client_timestamp': self._extract_timestamp(start_events['timestamps'], start_idx, 'client'),
                                'server_timestamp': self._extract_timestamp(start_events['timestamps'], start_idx, 'server')
                            }

                            end_data = {
                                'data': end_events['data'][end_idx] if end_idx < len(end_events['data']) else {},
                                'event_ids': [end_id],
                                'client_timestamp': self._extract_timestamp(end_events['timestamps'], end_idx, 'client'),
                                'server_timestamp': self._extract_timestamp(end_events['timestamps'], end_idx, 'server')
                            }

                            pairs[start_id] = (start_data, end_data)

        return pairs
    
    def _process_event_data(self, events, session_id, h5_file):
        """Process event data into tasks, elements, and segments.
        
        Args:
            events: Dictionary of events
            session_id: Session ID
            h5_file: Source H5 file
            
        Returns:
            dict: Processed event data
        """
        # Initialize result dictionaries
        tasks = {}
        elements = {}
        segments = {
            'recording': [],
            'thinking': [],
            'pause': []
        }
        
        # Get session start time
        session_start_time = self._get_session_start_time(h5_file)
        
        # 1. Process task events
        task_pairs = self._pair_events(events, 'task_started', 'task_completed')
        
        for event_id, (start_event, end_event) in task_pairs.items():
            # Extract task data from event
            task_data = start_event['data']
            config = task_data.get('config', {})
            
            # Extract actual task ID from the data (e.g., "eye_1" instead of "task_started_1")
            actual_task_id = task_data.get('task_id', '')
            
            if not actual_task_id:
                self.logger.warning(f"Task started event {event_id} has no task_id field")
                continue
                
            # Create task entry using the actual task ID as the key
            tasks[actual_task_id] = {
                # Core identifiers
                'task_id': actual_task_id,  # Use the actual task ID from data
                'task_type': task_data.get('task_type', ''),
                'start_time': start_event['client_timestamp'],
                'end_time': end_event['client_timestamp'],
                'completion_status': 'completed',
                # Store the original event IDs for reference
                'task_started_event_id': event_id,
                
                # Core configuration - common across task types
                'count': config.get('count', 0),
                'allow_repeats': config.get('allow_repeats', False),
                'with_interruptions': config.get('with_interruptions', False),
                'audio_mode': config.get('audio_mode') or DEFAULT_AUDIO_MODE,
                
                
                # Input configuration parameters - variable by task type
                'input_modality': config.get('input_modality') or DEFAULT_INPUT_MODALITY,
                
                
                # Event references
                'task_started_id': start_event['event_ids'][0],
                'task_completed_id': end_event['event_ids'][0],
                
                # Skip information (populated later if applicable)
                'skipped': False,
                'skip_time': 0.0,
                'skip_event_id': '',
                
                # Element count will be calculated, no need to store element IDs
            }
            
            # Get task reference to reduce redundant dictionary lookups
            # print(f"DEBUG-TASK: Adding task_id '{actual_task_id}' to dictionary")
            task = tasks[actual_task_id]

            # Calculate additional metadata
            task['duration'] = task['end_time'] - task['start_time']

            # Calculate session fractions
            if session_start_time is not None:
                task['session_fraction_start'] = (task['start_time'] - session_start_time) / (24 * 60 * 60)
                task['session_fraction_end'] = (task['end_time'] - session_start_time) / (24 * 60 * 60)
        
        # 2. Process skip task events if present
        if 'skip_task' in events and len(events['skip_task']['event_ids']) > 0:
            for i, skip_id in enumerate(events['skip_task']['event_ids']):
                if i >= len(events['skip_task']['timestamps']):
                    continue
                    
                # NOTE: In H5 files, timestamps are [server_timestamp, client_timestamp]
                skip_time = float(events['skip_task']['timestamps'][i][1])  # Use client timestamp
                
                # Find active task at skip time
                active_task_id = None
                for t_id, task in tasks.items():
                    if task['start_time'] <= skip_time <= task['end_time']:
                        active_task_id = t_id
                        break
                
                # If no active task found, find the next scheduled task
                if not active_task_id:
                    next_tasks = [(t_id, t['start_time']) for t_id, t in tasks.items() if t['start_time'] > skip_time]
                    if next_tasks:
                        active_task_id = min(next_tasks, key=lambda x: x[1])[0]
                
                # Update task if found
                if active_task_id and active_task_id in tasks:
                    # Get task reference
                    task = tasks[active_task_id]

                    task['skipped'] = True
                    task['skip_event_id'] = skip_id
                    task['skip_time'] = skip_time

                    # Update status based on when the skip occurred
                    if skip_time <= task['start_time']:
                        task['completion_status'] = 'skipped_before_start'
                    else:
                        task['completion_status'] = 'partially_completed_then_skipped'
        
        # 3. Process element events
        element_pairs = self._pair_events(events, 'element_sent', 'element_replied')
        
        for event_id, (sent_event, replied_event) in element_pairs.items():
            # Extract element data
            sent_data = sent_event['data']
            replied_data = replied_event['data'] if replied_event else {}

            element_content = sent_data.get('element_content', {})
            task_metadata = element_content.get('task_metadata', {})

            # Extract the actual element_id from the data, fallback to event_id if not found
            # First check in sent_data, then in element_content
            actual_element_id = sent_data.get('element_id') or element_content.get('element_id', event_id)

            # DEBUG: Print timestamps before element creation
            print(f"DEBUG-TS-ELEMENT-CREATE: Creating element '{actual_element_id}'")
            # print(f"DEBUG-TS-ELEMENT-CREATE: sent_event client timestamp: {sent_event['client_timestamp']}")
            # print(f"DEBUG-TS-ELEMENT-CREATE: sent_event server timestamp: {sent_event['server_timestamp']}")
            # print(f"DEBUG-TS-ELEMENT-CREATE: replied_event client timestamp: {replied_event['client_timestamp']}")
            # print(f"DEBUG-TS-ELEMENT-CREATE: replied_event server timestamp: {replied_event['server_timestamp']}")

            # Create element entry with extracted fields
            elements[actual_element_id] = {
                'element_id': actual_element_id,
                'element_type': element_content.get('element_type', ''),
                'title': element_content.get('title', ''),
                'is_instruction': element_content.get('is_instruction', False),

                # Task relationship
                'task_id': task_metadata.get('task_id', ''),
                'task_type': task_metadata.get('task_type', ''),
                'sequence_idx': task_metadata.get('current_count', 0),
                'max_count': task_metadata.get('max_count', 0),

                # Timing
                'start_time': sent_event['client_timestamp'],
                'end_time': replied_event['client_timestamp'],
                'session_fraction': task_metadata.get('fraction_session_completed', 0.0),
                
                # Presentation
                'audio_mode': element_content.get('audio_mode') or DEFAULT_AUDIO_MODE,
                'with_interruptions': element_content.get('with_interruptions', False),
                
                # Response
                'input_modality': replied_data.get('input_modality', ''),
                
                # Event references - keep original event IDs for reference
                'element_sent_id': sent_event['event_ids'][0],
                'element_replied_id': replied_event['event_ids'][0],
                'event_id': event_id,  # Store the original pairing ID for reference
                
                # No segment placeholders needed
            }
            
            # Get element reference to reduce redundant dictionary lookups
            element = elements[actual_element_id]

            # Calculate derived fields
            element['duration'] = element['end_time'] - element['start_time']

            # Get configured response time limit from element_content or task config
            # First check directly in element_content
            # Then check in task_metadata's config if present
            # Default to -1 if not configured
            response_config = task_metadata.get('config', {})
            element['response_time_seconds'] = (
                element_content.get('response_time_seconds') or
                response_config.get('response_time_seconds') or
                -1
            )

            if session_start_time is not None:
                element['session_relative_time'] = element['start_time'] - session_start_time

            # Add element to task's element list
            task_id = element['task_id']
            # print(f"DEBUG-ELEMENT: Element {actual_element_id} looking for task_id '{task_id}'")
            if task_id and task_id in tasks:
                # Get task reference
                task = tasks[task_id]

                # Instead of storing element IDs directly, just increment the count
                if 'element_count' not in task:
                    task['element_count'] = 0
                task['element_count'] += 1
            else:
                self.logger.warning(f"Element {actual_element_id} could not be associated with any task (task_id='{task_id}')")
        
        # 4. Process segment events
        segment_types = [
            ('recording', 'recording_start', 'recording_stop'),
            ('thinking', 'thinking_start', 'thinking_stop'),
            ('pause', 'pause_event', 'resume_event')
        ]
        
        # Process all segment types using a consistent approach
        for segment_type, start_event_type, end_event_type in segment_types:
            self._process_segment_type(events, start_event_type, end_event_type, segment_type, segments)
        
        # 5. Associate segments with elements and tasks
        # For each segment type
        for segment_type, segment_list in segments.items():
            for segment in segment_list:
                # Get timestamps
                start_time = segment['start_time']
                end_time = segment['end_time']

                # DEBUG: Print segment information
                # print(f"DEBUG-SEGMENT: Processing {segment_type} segment '{segment['segment_id']}' with start_time={start_time}")

                # Check element containment - only check if start time is within element timespan
                found_containing_element = False
                for element_id, element in elements.items():
                    # DEBUG: Print element time bounds being checked
                    # print(f"DEBUG-ELEMENT-TIMES: Checking element '{element_id}' with start={element['start_time']}, end={element['end_time']}")

                    # Check if segment is contained in this element
                    if self._is_segment_contained_in_element(start_time, element, segment['segment_id'], element_id):
                        # Add reference only to segment - no bidirectional references
                        segment['containing_element_id'] = element_id

                        # Calculate relative position
                        segment['element_relative_start'] = start_time - element['start_time']

                        # DEBUG: Print successful containment
                        # print(f"DEBUG-CONTAINMENT: Segment '{segment['segment_id']}' is contained by element '{element_id}'")
                        found_containing_element = True
                        break

                # DEBUG: Print warning if no element contains this segment
                if not found_containing_element:
                    print(f"DEBUG-CONTAINMENT-WARNING: Segment '{segment['segment_id']}' is not contained by any element!")

                # No fallback to task containment - segments are only contained by elements
        
        return {
            'tasks': tasks,
            'elements': elements,
            'segments': segments,
            'session_id': session_id
        }
    
    def _get_session_start_time(self, h5_file):
        """Get the session start time from metadata.
        
        Args:
            h5_file: H5 file object
            
        Returns:
            float: Session start time or None
        """
        if 'metadata' in h5_file and 'start_time' in h5_file['metadata'].attrs:
            return float(h5_file['metadata'].attrs['start_time'])
        return None
    
    @classmethod
    def from_args(cls, args):
        """Create instance from command line arguments.
        
        Args:
            args: Command line arguments
            
        Returns:
            EventTransform: Instance of transform
        """
        # Extract arguments
        source_prefix = getattr(args, 'source_prefix', cls.SOURCE_PREFIX)
        dest_prefix = getattr(args, 'dest_prefix', cls.DEST_PREFIX)
        
        kwargs = {
            'source_prefix': source_prefix,
            'destination_prefix': dest_prefix,
            's3_bucket': args.s3_bucket,
            'verbose': args.verbose,
            'log_file': args.log_file,
            'dry_run': args.dry_run
        }
        
        # Handle keep_local if it exists
        if hasattr(args, 'keep_local'):
            kwargs['keep_local'] = args.keep_local
            
        return cls(**kwargs)
    
    @classmethod
    def add_subclass_arguments(cls, parser):
        """Add subclass-specific arguments to parser.
        
        Args:
            parser: ArgumentParser to add arguments to
        """
        # No additional arguments needed for basic operation
        pass


if __name__ == "__main__":
    EventTransform.run_from_command_line()