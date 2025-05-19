"""
Language tokenization transform for windowed H5 files.

This transform:
1. Reads windowed H5 files from S3 (processed/windows/)
2. Extracts language data from source curated H5 files
3. Tokenizes the text using a specified tokenizer
4. Saves results to processed/lang/ prefix in S3
5. Records metadata in DynamoDB

This is implemented using the new BaseTransform architecture.
"""

import os
import sys
import json
import time
import h5py
import numpy as np
from typing import Dict, Any, List, Optional

# Import base transform
from base_transform import BaseTransform, Session

# Import transformers tokenizer
from transformers import AutoTokenizer

# Import tokenization functions
from .lang_processing.tokenization import extract_and_tokenize, extract_and_tokenize_words
from .lang_processing.correction import correct_and_align_w_group


class LangTransform(BaseTransform):
    """
    Language tokenization transform for neural data.

    This transform processes language data through several stages:
    1. Extracts language/L (listen) and language/W (written) character data
    2. Tokenizes the text using a specified tokenizer
    3. Stores tokenized data and metadata
    """

    # Define required class attributes for source and destination
    SOURCE_PREFIX = 'curated-h5/'
    DEST_PREFIX = 'processed/lang/'

    def __init__(self, tokenizer_name: str = 'gpt2',
                 **kwargs):
        """
        Initialize the language tokenization transform.

        Args:
            tokenizer_name: Name of the tokenizer to use (e.g., gpt2, facebook/opt-125m)
            **kwargs: Additional arguments for BaseTransform
        """
        # Set default transform info if not provided
        transform_id = kwargs.pop('transform_id', 'lang_v0')
        script_id = kwargs.pop('script_id', '2B')
        script_name = kwargs.pop('script_name', 'tokenize_language')
        script_version = kwargs.pop('script_version', 'v0')

        # Call parent constructor
        super().__init__(
            transform_id=transform_id,
            script_id=script_id,
            script_name=script_name,
            script_version=script_version,
            **kwargs
        )

        # Set lang-specific attributes
        self.tokenizer_name = tokenizer_name
        self._tokenizer = None

        self.logger.info(f"Language tokenization transform initialized with:")
        self.logger.info(f"  Tokenizer: {self.tokenizer_name}")

    @property
    def tokenizer(self):
        """Lazy-load tokenizer when needed."""
        if self._tokenizer is None:
            self.logger.info(f"Loading tokenizer: {self.tokenizer_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        return self._tokenizer

    def process_session(self, session: Session) -> Dict:
        """Process a single session.

        This implementation:
        1. Finds the curated H5 file for the session
        2. Extracts language data
        3. Tokenizes the text
        4. Saves the tokenized data to a zarr store

        Args:
            session: Session object

        Returns:
            Dict with processing results
        """
        session_id = session.session_id
        self.logger.info(f"Processing session: {session_id}")

        # In curated-h5/, files are always directly in the source prefix
        curated_h5_key = f"{self.source_prefix}{session_id}.h5"

        # Check if the file exists
        try:
            self.s3.head_object(Bucket=self.s3_bucket, Key=curated_h5_key)
            self.logger.info(f"Found H5 file: {curated_h5_key}")
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
        local_h5_path = session.download_file(curated_h5_key)

        try:
            # Analyze the file to determine if it has language data
            analysis = self.analyze_file(local_h5_path, session_id)
            
            # If there's no language data to process, skip
            if not analysis.get('qualifies', False):
                self.logger.info(f"No language data to process for session {session_id}")
                return {
                    "status": "skipped",
                    "metadata": analysis,
                    "files_to_copy": [],
                    "files_to_upload": [],
                    "zarr_stores": []
                }
            
            # Process the file to extract and tokenize language data
            self.logger.info(f"Processing session {session_id} with language data")
            
            # Process language data
            result = self.process_language_data(local_h5_path, session, analysis)
            
            # Create metadata
            metadata = {
                "session_id": session_id,
                "tokenizer": self.tokenizer_name,
                "has_L_data": analysis.get('has_L_data', False),
                "has_W_data": analysis.get('has_W_data', False),
                "has_R_data": analysis.get('has_R_data', False),
                "has_S_data": analysis.get('has_S_data', False),
                "has_W_corrected": analysis.get('has_W_data', False),
                "L_chars_count": analysis.get('L_chars_count', 0),
                "W_chars_count": analysis.get('W_chars_count', 0),
                "R_words_count": analysis.get('R_words_count', 0),
                "S_words_count": analysis.get('S_words_count', 0),
                "W_corrected_token_count": result.get('details', {}).get('W_corrected_token_count', 0),
                "W_correctness_score": result.get('details', {}).get('W_correctness_score', 0),
                "storage_format": "zarr3",  # Add this line to indicate zarr format
                "processing_details": result.get('details', {})
            }
            
            # Get the store key from the result
            store_key = result.get('store_key')
            
            if not store_key:
                self.logger.warning(f"No zarr store created for session {session_id}")
                return {
                    "status": "failed",
                    "error_details": "Failed to create zarr store",
                    "metadata": metadata,
                    "files_to_copy": [],
                    "files_to_upload": [],
                    "zarr_stores": []
                }
            
            return {
                "status": "success",
                "metadata": metadata,
                "files_to_copy": [],
                "files_to_upload": [],
                "zarr_stores": [store_key]
            }
            
        except Exception as e:
            self.logger.error(f"Error processing session {session_id}: {e}", exc_info=True)
            return {
                "status": "failed",
                "error_details": str(e),
                "metadata": {"session_id": session_id},
                "files_to_copy": [],
                "files_to_upload": [],
                "zarr_stores": []
            }
        finally:
            # Only cleanup if not keep_local
            if not self.keep_local and 'local_h5_path' in locals():
                if os.path.exists(local_h5_path):
                    try:
                        os.remove(local_h5_path)
                        self.logger.debug(f"Removed temporary file {local_h5_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to remove temporary file {local_h5_path}: {e}")

    def analyze_file(self, file_path: str, session_id: str) -> Dict:
        """Analyze H5 file to determine if it has language data.
        
        Args:
            file_path: Path to the local H5 file
            session_id: Session ID
            
        Returns:
            Dict with analysis results
        """
        # Initialize analysis results
        analysis = {
            'session_id': session_id,
            'has_L_data': False,
            'has_W_data': False,
            'has_R_data': False,  # Add R data flag
            'has_S_data': False,  # Add S data flag
            'L_chars_count': 0,
            'W_chars_count': 0,
            'R_words_count': 0,   # Add R words count
            'S_words_count': 0,   # Add S words count
            'qualifies': False
        }
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Check if file has language group
                if 'language' not in f:
                    self.logger.info(f"No 'language' group found in {session_id}")
                    return analysis
                
                # Check for L and W subgroups with chars datasets
                if 'L' in f['language'] and 'chars' in f['language']['L']:
                    analysis['has_L_data'] = True
                    analysis['L_chars_count'] = len(f['language/L/chars'])
                    self.logger.info(f"Found 'L' data with {analysis['L_chars_count']} characters")
                
                if 'W' in f['language'] and 'chars' in f['language']['W']:
                    analysis['has_W_data'] = True
                    analysis['W_chars_count'] = len(f['language/W/chars'])
                    self.logger.info(f"Found 'W' data with {analysis['W_chars_count']} characters")
                
                # Check for R and S subgroups with words datasets
                if 'R' in f['language'] and 'words' in f['language']['R']:
                    analysis['has_R_data'] = True
                    analysis['R_words_count'] = len(f['language/R/words'])
                    self.logger.info(f"Found 'R' data with {analysis['R_words_count']} words")
                
                if 'S' in f['language'] and 'words' in f['language']['S']:
                    analysis['has_S_data'] = True
                    analysis['S_words_count'] = len(f['language/S/words'])
                    self.logger.info(f"Found 'S' data with {analysis['S_words_count']} words")
                
                # The file qualifies if it has any language data (L, W, R, or S)
                qualifies = (analysis['has_L_data'] or analysis['has_W_data'] or 
                              analysis['has_R_data'] or analysis['has_S_data'])
                analysis['qualifies'] = qualifies
                
                if qualifies:
                    self.logger.info(f"Session {session_id} has language data to tokenize")
                else:
                    self.logger.info(f"Session {session_id} doesn't have language data")
        
        except Exception as e:
            self.logger.error(f"Error analyzing file {session_id}: {e}")
            analysis['error'] = str(e)
        
        return analysis

    def _convert_tokens_to_arrays(self, tokens_data, group_name):
        """
        Convert tokens list to structured numpy arrays.
        
        Args:
            tokens_data: List of token dictionaries
            group_name: Name of the language group (L, W, W_corrected, R, S)
            
        Returns:
            Dict of arrays for each token attribute
        """
        import numpy as np
        
        # Common token attributes for all groups
        token_arrays = {
            'token': [],
            'token_id': [],
            'start_timestamp': [],
            'end_timestamp': [],
            'special_token': []
        }
        
        # Add group-specific arrays
        if group_name == 'W_corrected':
            token_arrays['original_indices'] = []
            token_arrays['unchanged'] = []
        elif group_name == 'W':
            token_arrays['keystroke_events'] = []
            token_arrays['trigger_keystrokes'] = []
            token_arrays['char_indices'] = []
        
        # Extract values from tokens
        for token in tokens_data:
            # Common attributes
            token_arrays['token'].append(token.get('token', ''))
            token_arrays['token_id'].append(token.get('token_id', 0))
            token_arrays['start_timestamp'].append(token.get('start_timestamp', 0.0))
            token_arrays['end_timestamp'].append(token.get('end_timestamp', 0.0))
            token_arrays['special_token'].append(token.get('special_token', False))
            
            # Group-specific attributes
            if group_name == 'W_corrected':
                token_arrays['original_indices'].append(np.array(token.get('original_indices', []), dtype=np.int32))
                token_arrays['unchanged'].append(token.get('unchanged', False))
            elif group_name == 'W':
                token_arrays['keystroke_events'].append(token.get('keystroke_events', json.dumps([])))
                token_arrays['trigger_keystrokes'].append(token.get('trigger_keystrokes', json.dumps([])))
                token_arrays['char_indices'].append(np.array(token.get('char_indices', []), dtype=np.int32))
        
        # Convert lists to numpy arrays
        result = {}
        for key, values in token_arrays.items():
            # Handle special array types
            if key in ['original_indices', 'char_indices']:
                # These are already numpy arrays
                result[key] = values
            elif key in ['token', 'keystroke_events', 'trigger_keystrokes']:
                # Convert to fixed-length strings
                max_len = max(1, max(len(str(v)) for v in values))
                result[key] = np.array(values, dtype=f'U{max_len}')
            elif key == 'token_id':
                # Integer arrays
                result[key] = np.array(values, dtype=np.int32)
            elif key in ['special_token', 'unchanged']:
                # Boolean arrays
                result[key] = np.array(values, dtype=bool)
            else:
                # Default to float64 for timestamps
                result[key] = np.array(values, dtype=np.float64)
        
        return result

    def process_language_data(self, file_path: str, session: Session, analysis: Dict) -> Dict:
        """Process language data and create tokenized output.
        
        Args:
            file_path: Path to the local H5 file
            session: Session object
            analysis: Analysis results
            
        Returns:
            Dict with processing details
        """
        session_id = session.session_id
        self.logger.info(f"Processing language data for session {session_id}")
        
        tokenization_results = {}
        processing_details = {}
        
        try:
            # Process the file
            with h5py.File(file_path, 'r') as f_in:
                # Process L and W groups if they exist
                if analysis['has_L_data']:
                    result = extract_and_tokenize(f_in, 'L', self.tokenizer)
                    if result:
                        tokenization_results['L'] = result
                        token_count = len(result['tokens'])
                        char_count = len(result['text'])
                        self.logger.info(f"Tokenized {char_count} chars into {token_count} tokens for L group")
                        processing_details['L_token_count'] = token_count
                        processing_details['L_char_count'] = char_count
                
                if analysis['has_W_data']:
                    # Extract character data first (only once)
                    chars_dataset = f_in[f'language/W/chars']
                    chars_data = []
                    
                    for i in range(len(chars_dataset)):
                        char_record = chars_dataset[i]
                        char = char_record['char']
                        if isinstance(char, bytes):
                            char = char.decode('utf-8')
                        
                        # Extract all fields including keystroke data
                        data = {
                            'char': char,
                            'timestamp': char_record['timestamp'],
                            'idx': i
                        }
                        
                        # Add keystroke data if available in the dataset
                        if 'keystrokes' in char_record.dtype.names:
                            data['keystrokes'] = char_record['keystrokes']
                        if 'trigger_keystroke' in char_record.dtype.names:
                            data['trigger_keystroke'] = char_record['trigger_keystroke']
                        if 'reconstructed' in char_record.dtype.names:
                            data['reconstructed'] = char_record['reconstructed']
                        
                        chars_data.append(data)
                    
                    # Process W group normally with the existing functionality
                    result = extract_and_tokenize(f_in, 'W', self.tokenizer)
                    if result:
                        tokenization_results['W'] = result
                        token_count = len(result['tokens'])
                        char_count = len(result['text'])
                        self.logger.info(f"Tokenized {char_count} chars into {token_count} tokens for W group")
                        processing_details['W_token_count'] = token_count
                        processing_details['W_char_count'] = char_count
                        
                        # Add spelling correction for W group
                        self.logger.info(f"Applying spelling correction to W group")
                        
                        # Apply correction and get aligned tokens
                        corrected_result, correctness_score = correct_and_align_w_group(chars_data, self.tokenizer)
                        
                        # Add results to tokenization_results
                        tokenization_results['W_corrected'] = corrected_result
                        
                        # Record metrics
                        processing_details['W_corrected_token_count'] = len(corrected_result['tokens'])
                        processing_details['W_correctness_score'] = correctness_score
                        
                        self.logger.info(f"Applied spelling correction with correctness score: {correctness_score:.2f}")
                
                # Process R and S groups if they exist
                if analysis.get('has_R_data', False):
                    result = extract_and_tokenize_words(f_in, 'R', self.tokenizer)
                    if result:
                        tokenization_results['R'] = result
                        token_count = len(result['tokens'])
                        word_count = len(result['text'].split())
                        self.logger.info(f"Tokenized {word_count} words into {token_count} tokens for R group")
                        processing_details['R_token_count'] = token_count
                        processing_details['R_word_count'] = word_count
                
                if analysis.get('has_S_data', False):
                    result = extract_and_tokenize_words(f_in, 'S', self.tokenizer)
                    if result:
                        tokenization_results['S'] = result
                        token_count = len(result['tokens'])
                        word_count = len(result['text'].split())
                        self.logger.info(f"Tokenized {word_count} words into {token_count} tokens for S group")
                        processing_details['S_token_count'] = token_count
                        processing_details['S_word_count'] = word_count
            
            # If nothing was processed, return early
            if not tokenization_results:
                self.logger.warning(f"No language data processed for {session_id}")
                return {"details": processing_details}
            
            # Calculate total stats
            total_token_count = sum(len(r['tokens']) for r in tokenization_results.values())
            char_groups = {k: v for k, v in tokenization_results.items() if k in ['L', 'W']}
            word_groups = {k: v for k, v in tokenization_results.items() if k in ['R', 'S']}
            
            total_char_count = sum(len(r['text']) for r in char_groups.values())
            total_word_count = sum(len(r['text'].split()) for r in word_groups.values())
            
            processing_details['total_token_count'] = total_token_count
            processing_details['total_char_count'] = total_char_count
            processing_details['total_word_count'] = total_word_count
            
            # Build metadata dictionary
            metadata = {
                'tokenizer': self.tokenizer_name,
                'tokenization_date': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                'has_L_data': analysis.get('has_L_data', False),
                'has_W_data': analysis.get('has_W_data', False),
                'has_R_data': analysis.get('has_R_data', False),
                'has_S_data': analysis.get('has_S_data', False),
                'has_W_corrected': analysis.get('has_W_data', False),
                'L_chars_count': analysis.get('L_chars_count', 0),
                'W_chars_count': analysis.get('W_chars_count', 0),
                'R_words_count': analysis.get('R_words_count', 0),
                'S_words_count': analysis.get('S_words_count', 0),
                'W_correctness_score': processing_details.get('W_correctness_score', 0.0),
                'session_id': session_id,
                'storage_format': 'zarr3'
            }
            
            # Build zarr tree structure
            zarr_tree = {
                'language': {},
                'metadata': metadata
            }
            
            # Process each language group
            for group_name, result in tokenization_results.items():
                # Skip empty groups
                if not result or 'tokens' not in result or not result['tokens']:
                    continue
                
                # Create array for tokens
                token_arrays = self._convert_tokens_to_arrays(result['tokens'], group_name)
                
                # Add to zarr tree
                zarr_tree['language'][group_name] = {
                    **token_arrays,
                    'text': result['text']
                }
                
                # Add special attributes for W_corrected
                if group_name == 'W_corrected':
                    zarr_tree['language'][group_name]['original_text'] = tokenization_results['W']['text']
                    zarr_tree['language'][group_name]['correctness_score'] = result.get('correctness_score', 0.0)
            
            # Define zarr store key with tokenizer name
            store_key = f"{self.destination_prefix}{session_id}_{self.tokenizer_name}_lang.zarr"
            
            # Set root attributes
            root_attrs = {
                'session_id': session_id,
                'tokenizer': self.tokenizer_name,
                'transform': 'lang',
                'version': '0.1',
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                'storage_format': 'zarr3'
            }
            
            # Create zarr store using the BaseTransform helper
            if not self.dry_run:
                self.save_zarr_dict_to_s3(store_key, zarr_tree, root_attrs)
            else:
                self.logger.info(f"[DRY RUN] Would create zarr store at {store_key}")
            
            return {
                "store_key": store_key,
                "details": processing_details
            }
        
        except Exception as e:
            self.logger.error(f"Error processing language data for {session_id}: {e}", exc_info=True)
            return {"details": {"error": str(e)}}

    @classmethod
    def add_subclass_arguments(cls, parser):
        """Add language tokenization-specific command-line arguments.
        
        Args:
            parser: ArgumentParser instance
        """
        parser.add_argument('--tokenizer', type=str, default='gpt2',
                          help='Name of the tokenizer to use (e.g., gpt2, facebook/opt-125m)')

    @classmethod
    def from_args(cls, args):
        """Create a transform instance from command-line arguments.
        
        Args:
            args: Parsed command-line arguments
            
        Returns:
            Instance of LangTransform
        """
        # Extract arguments
        source_prefix = getattr(args, 'source_prefix', cls.SOURCE_PREFIX)
        dest_prefix = getattr(args, 'dest_prefix', cls.DEST_PREFIX)
        
        return cls(
            source_prefix=source_prefix,
            destination_prefix=dest_prefix,
            tokenizer_name=args.tokenizer,
            s3_bucket=args.s3_bucket,
            verbose=args.verbose,
            log_file=args.log_file,
            dry_run=args.dry_run,
            keep_local=args.keep_local
        )


# Entry point for running the transform from the command line
if __name__ == "__main__":
    LangTransform.run_from_command_line()