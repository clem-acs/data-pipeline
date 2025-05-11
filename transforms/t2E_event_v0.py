"""
T2E Events Transform

Extracts and organizes event data from curated H5 files into a structured format
for easy integration with TileDB and other transforms.
"""

import os
import json
import logging
import numpy as np
import h5py
import time
from typing import Dict, Any, List, Optional
from collections import defaultdict

# Import base transform
from base_transform import BaseTransform, Session

# Constants for default values
DEFAULT_AUDIO_MODE = "text_only"
DEFAULT_INPUT_MODALITY = "text"

# Module logger - only used before class initialization
logger = logging.getLogger(__name__)

class EventTransform(BaseTransform):
    """Transform for extracting and organizing event data from H5 files."""
    
    SOURCE_PREFIX = "curated-h5/"
    DEST_PREFIX = "processed/event/"
    
    def __init__(self, **kwargs):
        """
        Initialize the events transform.

        Args:
            **kwargs: Additional arguments for BaseTransform
        """
        # Set default transform info if not provided
        transform_id = kwargs.pop('transform_id', 't2E_event_v0')
        script_id = kwargs.pop('script_id', '2E')
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
                    "files_to_upload": []
                }
            
            # Download the H5 file
            local_source = session.download_file(h5_key)
            
            # Create output file
            output_filename = f"{session_id}_events.h5"
            local_dest = session.create_upload_file(output_filename)
            
            # Process the file
            with h5py.File(local_source, 'r') as source_h5:
                # Extract and process events
                self.logger.info(f"Extracting events for {session_id}")
                success = self._process_events(source_h5, local_dest, session_id)
                
                if not success:
                    self.logger.error(f"Failed to process events for {session_id}")
                    return {
                        "status": "failed",
                        "error_details": f"Failed to process events for {session_id}",
                        "metadata": {"session_id": session_id},
                        "files_to_copy": [],
                        "files_to_upload": []
                    }
            
            # Create metadata for DynamoDB
            metadata = {
                "session_id": session_id,
                "processed_at": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            }
            
            # Define the destination key
            dest_key = f"{self.destination_prefix}{output_filename}"
            
            self.logger.info(f"Successfully processed events for {session_id}")
            return {
                "status": "success",
                "metadata": metadata,
                "files_to_copy": [],
                "files_to_upload": [(local_dest, dest_key)]
            }
            
        except Exception as e:
            self.logger.error(f"Error processing events for {session_id}: {e}", exc_info=True)
            return {
                "status": "failed",
                "error_details": str(e),
                "metadata": {"session_id": session_id},
                "files_to_copy": [],
                "files_to_upload": []
            }
    
    def _process_events(self, source_h5, dest_file, session_id):
        """Extract and organize events from H5 file.
        
        Args:
            source_h5: Source H5 file object
            dest_file: Path to destination file
            session_id: Session ID
            
        Returns:
            bool: True if processing succeeded
        """
        try:
            # Extract raw events from H5
            events = self._extract_raw_events(source_h5)
            
            # Process event data
            processed_data = self._process_event_data(events, session_id, source_h5)
            
            # Save to destination file
            self._save_processed_data(processed_data, dest_file)
            
            return True
        except Exception as e:
            self.logger.exception(f"Error processing events: {e}")
            return False
    
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
            start_time = float(start_event['timestamps'][0])
            end_time = float(end_event['timestamps'][0])
            
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
        
        # Simple matching by index if counts match
        if len(start_events['event_ids']) == len(end_events['event_ids']):
            for i in range(len(start_events['event_ids'])):
                start_id = start_events['event_ids'][i]
                end_id = end_events['event_ids'][i]
                
                start_data = {
                    'data': start_events['data'][i] if i < len(start_events['data']) else {},
                    'event_ids': [start_id],
                    'timestamps': start_events['timestamps'][i] if i < len(start_events['timestamps']) else []
                }
                
                end_data = {
                    'data': end_events['data'][i] if i < len(end_events['data']) else {},
                    'event_ids': [end_id],
                    'timestamps': end_events['timestamps'][i] if i < len(end_events['timestamps']) else []
                }
                
                pairs[start_id] = (start_data, end_data)
        else:
            # More complex matching by timestamps
            self.logger.warning(f"Mismatched counts for {start_type}/{end_type}, using timestamp matching")
            
            # Sort by timestamp
            start_items = sorted(list(zip(
                start_events['event_ids'],
                start_events['timestamps'][:, 0] if len(start_events['timestamps']) > 0 else [],
                range(len(start_events['event_ids']))
            )), key=lambda x: x[1])
            
            end_items = sorted(list(zip(
                end_events['event_ids'],
                end_events['timestamps'][:, 0] if len(end_events['timestamps']) > 0 else [],
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
                            'timestamps': start_events['timestamps'][start_idx] if start_idx < len(start_events['timestamps']) else []
                        }
                        
                        end_data = {
                            'data': end_events['data'][end_idx] if end_idx < len(end_events['data']) else {},
                            'event_ids': [end_id],
                            'timestamps': end_events['timestamps'][end_idx] if end_idx < len(end_events['timestamps']) else []
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
                'sequence_number': int(task_data.get('sequence', 0)),  # Extract sequence number
                'start_time': float(start_event['timestamps'][0]),
                'end_time': float(end_event['timestamps'][0]),
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
            
            # Calculate additional metadata
            print(f"DEBUG-TASK: Adding task_id '{actual_task_id}' to dictionary")
            tasks[actual_task_id]['duration'] = tasks[actual_task_id]['end_time'] - tasks[actual_task_id]['start_time']
            
            # Calculate session fractions
            if session_start_time is not None:
                tasks[actual_task_id]['session_fraction_start'] = (tasks[actual_task_id]['start_time'] - session_start_time) / (24 * 60 * 60)
                tasks[actual_task_id]['session_fraction_end'] = (tasks[actual_task_id]['end_time'] - session_start_time) / (24 * 60 * 60)
        
        # 2. Process skip task events if present
        if 'skip_task' in events and len(events['skip_task']['event_ids']) > 0:
            for i, skip_id in enumerate(events['skip_task']['event_ids']):
                if i >= len(events['skip_task']['timestamps']):
                    continue
                    
                skip_time = float(events['skip_task']['timestamps'][i][0])
                
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
                    tasks[active_task_id]['skipped'] = True
                    tasks[active_task_id]['skip_event_id'] = skip_id
                    tasks[active_task_id]['skip_time'] = skip_time
                    
                    # Update status based on when the skip occurred
                    if skip_time <= tasks[active_task_id]['start_time']:
                        tasks[active_task_id]['completion_status'] = 'skipped_before_start'
                    else:
                        tasks[active_task_id]['completion_status'] = 'partially_completed_then_skipped'
        
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
                'start_time': float(sent_event['timestamps'][0]),
                'end_time': float(replied_event['timestamps'][0]),
                'session_fraction': task_metadata.get('fraction_session_completed', 0.0),
                
                # Presentation
                'audio_mode': element_content.get('audio_mode') or DEFAULT_AUDIO_MODE,
                'has_audio': element_content.get('has_audio', False),
                'with_interruptions': element_content.get('with_interruptions', False),
                
                # Response
                'input_modality': replied_data.get('input_modality', ''),
                
                # Event references - keep original event IDs for reference
                'element_sent_id': sent_event['event_ids'][0],
                'element_replied_id': replied_event['event_ids'][0],
                'event_id': event_id,  # Store the original pairing ID for reference
                
                # No segment placeholders needed
            }
            
            # Calculate derived fields
            elements[actual_element_id]['duration'] = (
                elements[actual_element_id]['end_time'] - elements[actual_element_id]['start_time']
            )

            # Get configured response time limit from element_content or task config
            # First check directly in element_content
            # Then check in task_metadata's config if present
            # Default to -1 if not configured
            response_config = task_metadata.get('config', {})
            elements[actual_element_id]['response_time_seconds'] = (
                element_content.get('response_time_seconds') or
                response_config.get('response_time_seconds') or
                -1
            )

            if session_start_time is not None:
                elements[actual_element_id]['session_relative_time'] = elements[actual_element_id]['start_time'] - session_start_time

            # Add element to task's element list
            task_id = elements[actual_element_id]['task_id']
            print(f"DEBUG-ELEMENT: Element {actual_element_id} looking for task_id '{task_id}'")
            if task_id and task_id in tasks:
                # Instead of storing element IDs directly, just increment the count
                if 'element_count' not in tasks[task_id]:
                    tasks[task_id]['element_count'] = 0
                tasks[task_id]['element_count'] += 1
            else:
                self.logger.warning(f"Element {element_id} could not be associated with any task (task_id='{task_id}')")
        
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
                
                # Check element containment - only check if start time is within element timespan
                for element_id, element in elements.items():
                    if (element['start_time'] <= start_time and
                        element['end_time'] >= start_time):
                        # Add reference only to segment - no bidirectional references
                        segment['containing_element_id'] = element_id

                        # Calculate relative position
                        segment['element_relative_start'] = start_time - element['start_time']

                        break

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
    
    def _save_processed_data(self, processed_data, dest_file):
        """Save processed data to H5 file.
        
        Args:
            processed_data: Processed event data
            dest_file: Path to destination file
        """
        with h5py.File(dest_file, 'w') as h5f:
            # Create metadata group
            metadata = h5f.create_group('metadata')
            metadata.attrs['session_id'] = processed_data['session_id']
            metadata.attrs['transform'] = 'event'
            metadata.attrs['version'] = '0.1'
            
            # Print debug summary
            print(f"DEBUG-SUMMARY: All task_ids in dictionary: {list(processed_data['tasks'].keys())}")
            print(f"DEBUG-SUMMARY: Total tasks: {len(processed_data['tasks'])}, Total elements: {len(processed_data['elements'])}")
            
            # Create statistics
            stats = metadata.create_group('stats')
            stats.attrs['num_tasks'] = len(processed_data['tasks'])
            stats.attrs['num_elements'] = len(processed_data['elements'])
            
            for segment_type, segments in processed_data['segments'].items():
                stats.attrs[f'num_{segment_type}_segments'] = len(segments)
            
            # Create elements group and table
            elements_group = h5f.create_group('elements')
            element_dtype = self._create_element_dtype()
            element_data = self._convert_to_array(processed_data['elements'], element_dtype, 'elements')
            
            elements_table = elements_group.create_dataset(
                'table',
                data=element_data,
                dtype=element_dtype
            )
            
            # Create tasks group and table
            tasks_group = h5f.create_group('tasks')
            task_dtype = self._create_task_dtype()
            task_data = self._convert_to_array(processed_data['tasks'], task_dtype, 'tasks')
            
            tasks_table = tasks_group.create_dataset(
                'table',
                data=task_data,
                dtype=task_dtype
            )
            
            # Create segments group and tables
            segments_group = h5f.create_group('segments')
            segment_dtype = self._create_segment_dtype()
            
            for segment_type, segment_list in processed_data['segments'].items():
                if segment_list:
                    segment_data = self._convert_to_array(segment_list, segment_dtype, 'segments')
                    segments_group.create_dataset(
                        segment_type,
                        data=segment_data,
                        dtype=segment_dtype
                    )
            
            # Create indices
            indices_group = h5f.create_group('indices')
            
            # Create element_by_task index - build directly from elements
            element_by_task = indices_group.create_group('element_by_task')
            task_to_elements = defaultdict(list)

            # Collect elements by task_id
            for element_id, element in processed_data['elements'].items():
                task_id = element.get('task_id', '')
                if task_id:
                    task_to_elements[task_id].append(element_id)

            # Create datasets for each task's elements
            for task_id, element_ids in task_to_elements.items():
                if element_ids:
                    element_ids = [e_id.encode('utf-8') if isinstance(e_id, str) else e_id for e_id in element_ids]
                    element_by_task.create_dataset(
                        task_id,
                        data=np.array(element_ids, dtype='S64')
                    )
            
            # Element types can be queried directly from the elements table
            
            # Create segments_by_element index - built from segments table instead of element lists
            segments_by_element = indices_group.create_group('segments_by_element')
            element_segments = {}

            # Collect segments by their containing element
            for segment_type, segment_list in processed_data['segments'].items():
                for segment in segment_list:
                    element_id = segment.get('containing_element_id', '')
                    if element_id:
                        if element_id not in element_segments:
                            element_segments[element_id] = []
                        element_segments[element_id].append(segment['segment_id'])

            # Create datasets for each element's segments
            for element_id, segment_ids in element_segments.items():
                if segment_ids:
                    segment_ids = [s_id.encode('utf-8') if isinstance(s_id, str) else s_id for s_id in segment_ids]
                    segments_by_element.create_dataset(
                        element_id,
                        data=np.array(segment_ids, dtype='S64')
                    )
            
            # Segments by task can be derived by combining segments_by_element with element_by_task indices
    
    def _create_element_dtype(self):
        """Create numpy dtype for elements table.

        Returns:
            np.dtype: Element table dtype
        """
        return np.dtype([
            # 1. Identifiers & Metadata
            ('element_id', h5py.special_dtype(vlen=str)),
            ('element_type', h5py.special_dtype(vlen=str)),
            ('title', h5py.special_dtype(vlen=str)),
            ('is_instruction', np.bool_),

            # 2. Task Relationships
            ('task_id', h5py.special_dtype(vlen=str)),
            ('task_type', h5py.special_dtype(vlen=str)),
            ('sequence_idx', np.int32),
            ('max_count', np.int32),

            # 3. Timing & Position
            ('start_time', np.float64),
            ('end_time', np.float64),
            ('duration', np.float64),
            ('session_fraction', np.float32),
            ('session_relative_time', np.float64),

            # 4. Presentation Configuration
            ('audio_mode', h5py.special_dtype(vlen=str)),
            ('has_audio', np.bool_),
            ('with_interruptions', np.bool_),

            # 5. Response Characteristics
            ('input_modality', h5py.special_dtype(vlen=str)),
            ('response_time_seconds', np.float64),

            # 6. Event References
            ('element_sent_id', h5py.special_dtype(vlen=str)),
            ('element_replied_id', h5py.special_dtype(vlen=str)),
            ('event_id', h5py.special_dtype(vlen=str)),

            # Note: Segment relationships are now managed only through indices
        ])
    
    def _create_task_dtype(self):
        """Create numpy dtype for tasks table.

        Returns:
            np.dtype: Task table dtype
        """
        return np.dtype([
            # 1. Identifiers & Type
            ('task_id', h5py.special_dtype(vlen=str)),
            ('task_type', h5py.special_dtype(vlen=str)),
            ('sequence_number', np.int32),

            # 2. Timing
            ('start_time', np.float64),
            ('end_time', np.float64),
            ('duration', np.float64),
            ('session_fraction_start', np.float32),
            ('session_fraction_end', np.float32),

            # 3. Core Configuration
            ('count', np.int32),
            ('allow_repeats', np.bool_),
            ('with_interruptions', np.bool_),
            ('audio_mode', h5py.special_dtype(vlen=str)),


            # 5. Input Configuration
            ('input_modality', h5py.special_dtype(vlen=str)),

            # 6. Status Information
            ('completion_status', h5py.special_dtype(vlen=str)),
            ('skipped', np.bool_),
            ('skip_time', np.float64),
            ('skip_event_id', h5py.special_dtype(vlen=str)),

            # 7. Event References
            ('task_started_id', h5py.special_dtype(vlen=str)),
            ('task_completed_id', h5py.special_dtype(vlen=str)),

            # 8. Element Relationships
            ('element_count', np.int32),

            # Note: Segment relationships are now managed only through indices
        ])
    
    def _create_segment_dtype(self):
        """Create numpy dtype for segments table.
        
        Returns:
            np.dtype: Segment table dtype
        """
        return np.dtype([
            # 1. Core Information
            ('segment_id', h5py.special_dtype(vlen=str)),
            ('segment_type', h5py.special_dtype(vlen=str)),
            ('start_time', np.float64),
            ('end_time', np.float64),
            ('duration', np.float64),
            
            # 2. Container Relationships
            ('containing_element_id', h5py.special_dtype(vlen=str)),
            ('element_relative_start', np.float64),
            
            # 3. Event References
            ('start_event_id', h5py.special_dtype(vlen=str)),
            ('end_event_id', h5py.special_dtype(vlen=str)),
        ])
    
    def _convert_to_array(self, items, dtype, item_type='elements'):
        """Convert items to a structured numpy array with NULL safety.

        Args:
            items: Dictionary or list of items
            dtype: Numpy dtype for the array
            item_type: Type of items ('elements', 'tasks', or 'segments')

        Returns:
            np.ndarray: Structured array
        """
        # Initialize array
        array_data = np.zeros(len(items), dtype=dtype)

        # Define special fields that need array conversion by item type
        # Note: No more special fields requiring array conversion
        array_fields = {
            'elements': [],  # No segment arrays
            'tasks': [],     # No element_ids array
            'segments': []
        }
        special_fields = array_fields.get(item_type, [])

        # Convert items based on type
        if item_type == 'segments':
            # Segments handling (unchanged)
            for i, item in enumerate(items):
                for field in dtype.names:
                    if field in item:
                        array_data[i][field] = item[field]
        else:
            # Handle dictionary items (tasks, elements)
            for i, (item_id, item) in enumerate(items.items()):
                for field in dtype.names:
                    if field not in item:
                        # Initialize array fields to empty arrays instead of skipping
                        if field in special_fields:
                            array_data[i][field] = np.array([], dtype='S64')
                        continue

                    # Handle special field arrays with NULL safety
                    if field in special_fields:
                        # Always create a valid array, even if item[field] is None
                        if item[field] is None:
                            array_data[i][field] = np.array([], dtype='S64')
                        elif isinstance(item[field], list):
                            # Filter out None values and convert strings to bytes
                            ids = []
                            for id_val in item[field]:
                                if id_val is not None:
                                    if isinstance(id_val, str):
                                        ids.append(id_val.encode('utf-8'))
                                    else:
                                        ids.append(id_val)
                            array_data[i][field] = np.array(ids, dtype='S64')
                        else:
                            # Non-list value, create empty array
                            array_data[i][field] = np.array([], dtype='S64')
                    else:
                        # Regular non-array field
                        array_data[i][field] = item[field]

                # Task-specific post-processing
                if item_type == 'tasks':
                    # Element count is now directly maintained
                    if 'element_count' not in item:
                        array_data[i]['element_count'] = 0

                    # Set defaults for optional fields
                    if not item.get('completion_status'):
                        array_data[i]['completion_status'] = 'completed'

                    if 'skipped' not in item:
                        array_data[i]['skipped'] = False

        return array_data
    
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