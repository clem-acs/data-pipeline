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

        # Extract test and keep_local before passing kwargs to parent
        self.test = kwargs.pop('test', False)
        self.keep_local = kwargs.pop('keep_local', False)
        
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
        if self.test:
            self.logger.info(f"  TEST MODE: Will only extract metadata without tokenization")
        if self.keep_local:
            self.logger.info(f"  KEEP LOCAL: Will keep temporary files for inspection")

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
        4. Saves the tokenized data

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
                "files_to_upload": []
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
                    "files_to_upload": []
                }
            
            # Process the file to extract and tokenize language data
            self.logger.info(f"Processing session {session_id} with language data")
            
            # In test mode, just return the analysis
            if self.test:
                self.logger.info(f"TEST MODE: Skipping tokenization and file creation")
                return {
                    "status": "success",
                    "metadata": {**analysis, "test_mode": True},
                    "files_to_copy": [],
                    "files_to_upload": []
                }
            
            # Process language data
            result = self.process_language_data(local_h5_path, session, analysis)
            
            # Create metadata
            metadata = {
                "session_id": session_id,
                "tokenizer": self.tokenizer_name,
                "has_L_data": analysis.get('has_L_data', False),
                "has_W_data": analysis.get('has_W_data', False),
                "has_R_data": analysis.get('has_R_data', False),  # Add R data flag
                "has_S_data": analysis.get('has_S_data', False),  # Add S data flag
                "L_chars_count": analysis.get('L_chars_count', 0),
                "W_chars_count": analysis.get('W_chars_count', 0),
                "R_words_count": analysis.get('R_words_count', 0),  # Add R words count
                "S_words_count": analysis.get('S_words_count', 0),  # Add S words count
                "processing_details": result.get('details', {})
            }
            
            # Create the tokenized file
            tokenized_file_name = f"{session_id}_lang.h5"
            local_tokenized_path = result.get('local_path')
            
            if not local_tokenized_path:
                self.logger.warning(f"No tokenized file created for session {session_id}")
                return {
                    "status": "failed",
                    "error_details": "Failed to create tokenized file",
                    "metadata": metadata,
                    "files_to_copy": [],
                    "files_to_upload": []
                }
            
            # Define the destination key
            dest_key = f"{self.destination_prefix}{tokenized_file_name}"
            
            return {
                "status": "success",
                "metadata": metadata,
                "files_to_copy": [],
                "files_to_upload": [(local_tokenized_path, dest_key)]
            }
            
        except Exception as e:
            self.logger.error(f"Error processing session {session_id}: {e}", exc_info=True)
            return {
                "status": "failed",
                "error_details": str(e),
                "metadata": {"session_id": session_id},
                "files_to_copy": [],
                "files_to_upload": []
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

    def extract_and_tokenize(self, h5_file, group_name: str) -> Optional[Dict[str, Any]]:
        """Extract and tokenize text from a language group.
        
        Args:
            h5_file: Open h5py.File object
            group_name: 'L' or 'W' group name
            
        Returns:
            Dict with tokenization results or None
        """
        if f'language/{group_name}/chars' not in h5_file:
            return None
        
        # Extract characters
        chars_dataset = h5_file[f'language/{group_name}/chars']
        chars_data = []
        
        for i in range(len(chars_dataset)):
            char_record = chars_dataset[i]
            char = char_record['char']
            if isinstance(char, bytes):
                char = char.decode('utf-8')
            
            # Store character data with index 
            data = {'char': char, 'idx': i}
            
            # Handle timestamps based on group type
            if group_name == 'L':
                data['start_timestamp'] = char_record['start_timestamp']
                data['end_timestamp'] = char_record['end_timestamp']
            else:  # group_name == 'W'
                data['timestamp'] = char_record['timestamp']
            
            chars_data.append(data)
        
        if not chars_data:
            return None
        
        # Sort characters (important for W group which might not be in order)
        if group_name == 'L':
            chars_data.sort(key=lambda x: x['start_timestamp'])
        else:
            chars_data.sort(key=lambda x: x['timestamp'])
        
        # Regenerate text in correct order
        text = ''.join(c['char'] for c in chars_data)
        
        # Tokenize with offset mapping
        encoding = self.tokenizer(text, return_offsets_mapping=True)
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        token_ids = encoding['input_ids']
        offsets = encoding['offset_mapping']
        
        # Map tokens to characters
        token_data = []
        
        for token_idx, (start_offset, end_offset) in enumerate(offsets):
            # Handle special tokens
            if start_offset == end_offset:
                token_data.append({
                    'token': tokens[token_idx],
                    'token_id': token_ids[token_idx],
                    'start_timestamp': 0,
                    'end_timestamp': 0,
                    'special_token': True
                })
                continue
            
            # Find characters for this token
            chars_in_token = []
            for i in range(len(text)):
                if i >= start_offset and i < end_offset:
                    char_idx = i
                    if char_idx < len(chars_data):
                        chars_in_token.append(chars_data[char_idx])
            
            if chars_in_token:
                # Calculate token timestamps - different for L and W
                if group_name == 'L':
                    # For L: first char's start_timestamp to last char's end_timestamp
                    start_timestamp = chars_in_token[0]['start_timestamp']
                    end_timestamp = chars_in_token[-1]['end_timestamp']
                else:  # group_name == 'W'
                    # For W: first char's timestamp to last char's timestamp (single timestamps)
                    start_timestamp = chars_in_token[0]['timestamp']
                    end_timestamp = chars_in_token[-1]['timestamp']
                
                token_data.append({
                    'token': tokens[token_idx],
                    'token_id': token_ids[token_idx],
                    'start_timestamp': start_timestamp,
                    'end_timestamp': end_timestamp,
                    'special_token': False
                })
            else:
                # Should rarely happen, but handle just in case
                self.logger.warning(f"No characters found for token '{tokens[token_idx]}' at position {token_idx}")
                token_data.append({
                    'token': tokens[token_idx],
                    'token_id': token_ids[token_idx],
                    'start_timestamp': 0,
                    'end_timestamp': 0,
                    'special_token': False
                })
        
        return {'text': text, 'tokens': token_data}
        
    def extract_and_tokenize_words(self, h5_file, group_name: str) -> Optional[Dict[str, Any]]:
        """Extract and tokenize text from a word-level language group (R or S).
        
        Args:
            h5_file: Open h5py.File object
            group_name: 'R' or 'S' group name
            
        Returns:
            Dict with tokenization results or None
        """
        if f'language/{group_name}/words' not in h5_file:
            return None
        
        # Extract words
        words_dataset = h5_file[f'language/{group_name}/words']
        words_data = []
        
        for i in range(len(words_dataset)):
            word_record = words_dataset[i]
            word = word_record['word']
            if isinstance(word, bytes):
                word = word.decode('utf-8')
            
            # Store word data with timestamps
            words_data.append({
                'word': word,
                'start_timestamp': word_record['start_timestamp'],
                'end_timestamp': word_record['end_timestamp'],
                'idx': i
            })
        
        if not words_data:
            return None
        
        # Sort words by start_timestamp
        words_data.sort(key=lambda x: x['start_timestamp'])
        
        # Regenerate text by joining words with spaces
        text = ' '.join(w['word'] for w in words_data)
        
        # Precompute word positions in text (key optimization)
        words_with_positions = []
        current_pos = 0
        for word_data in words_data:
            word = word_data['word']
            word_start = current_pos
            word_end = word_start + len(word)
            words_with_positions.append({
                **word_data,
                'text_start': word_start, 
                'text_end': word_end
            })
            current_pos = word_end + 1  # +1 for space
        
        # Tokenize the text
        encoding = self.tokenizer(text, return_offsets_mapping=True)
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        token_ids = encoding['input_ids']
        offsets = encoding['offset_mapping']
        
        # Map tokens to words
        token_data = []
        
        for token_idx, (start_offset, end_offset) in enumerate(offsets):
            # Handle special tokens
            if start_offset == end_offset:
                token_data.append({
                    'token': tokens[token_idx],
                    'token_id': token_ids[token_idx],
                    'start_timestamp': 0,
                    'end_timestamp': 0,
                    'special_token': True
                })
                continue
            
            # Find words that overlap with this token (simple range check)
            related_words = [word_item for word_item in words_with_positions 
                             if word_item['text_start'] <= end_offset and word_item['text_end'] >= start_offset]
            
            if related_words:
                # Use timestamp from the first and last related word
                start_timestamp = related_words[0]['start_timestamp']
                end_timestamp = related_words[-1]['end_timestamp']
                
                token_data.append({
                    'token': tokens[token_idx],
                    'token_id': token_ids[token_idx],
                    'start_timestamp': start_timestamp,
                    'end_timestamp': end_timestamp,
                    'special_token': False
                })
            else:
                # If no related words found (unlikely), use default values
                self.logger.warning(f"No words found for token '{tokens[token_idx]}' at position {token_idx}")
                token_data.append({
                    'token': tokens[token_idx],
                    'token_id': token_ids[token_idx],
                    'start_timestamp': 0,
                    'end_timestamp': 0,
                    'special_token': False
                })
        
        return {'text': text, 'tokens': token_data}

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
                    result = self.extract_and_tokenize(f_in, 'L')
                    if result:
                        tokenization_results['L'] = result
                        token_count = len(result['tokens'])
                        char_count = len(result['text'])
                        self.logger.info(f"Tokenized {char_count} chars into {token_count} tokens for L group")
                        processing_details['L_token_count'] = token_count
                        processing_details['L_char_count'] = char_count
                
                if analysis['has_W_data']:
                    result = self.extract_and_tokenize(f_in, 'W')
                    if result:
                        tokenization_results['W'] = result
                        token_count = len(result['tokens'])
                        char_count = len(result['text'])
                        self.logger.info(f"Tokenized {char_count} chars into {token_count} tokens for W group")
                        processing_details['W_token_count'] = token_count
                        processing_details['W_char_count'] = char_count
                
                # Process R and S groups if they exist
                if analysis.get('has_R_data', False):
                    result = self.extract_and_tokenize_words(f_in, 'R')
                    if result:
                        tokenization_results['R'] = result
                        token_count = len(result['tokens'])
                        word_count = len(result['text'].split())
                        self.logger.info(f"Tokenized {word_count} words into {token_count} tokens for R group")
                        processing_details['R_token_count'] = token_count
                        processing_details['R_word_count'] = word_count
                
                if analysis.get('has_S_data', False):
                    result = self.extract_and_tokenize_words(f_in, 'S')
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
            
            # Create output file
            output_file_name = f"{session_id}_lang.h5"
            output_path = session.create_upload_file(output_file_name)
            
            with h5py.File(output_path, 'w') as f_out:
                lang_group = f_out.create_group('language')
                lang_group.attrs['tokenizer'] = self.tokenizer_name
                lang_group.attrs['session_id'] = session_id
                
                # Add metadata group
                meta_group = lang_group.create_group('metadata')
                meta_group.attrs['tokenizer'] = self.tokenizer_name
                meta_group.attrs['tokenization_date'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                meta_group.attrs['has_L_data'] = analysis.get('has_L_data', False)
                meta_group.attrs['has_W_data'] = analysis.get('has_W_data', False)
                meta_group.attrs['has_R_data'] = analysis.get('has_R_data', False)
                meta_group.attrs['has_S_data'] = analysis.get('has_S_data', False)
                meta_group.attrs['L_chars_count'] = analysis.get('L_chars_count', 0)
                meta_group.attrs['W_chars_count'] = analysis.get('W_chars_count', 0)
                meta_group.attrs['R_words_count'] = analysis.get('R_words_count', 0)
                meta_group.attrs['S_words_count'] = analysis.get('S_words_count', 0)
                
                # Add each tokenized group
                for group_name, result in tokenization_results.items():
                    group = lang_group.create_group(group_name)
                    
                    # Store original text as attribute
                    group.attrs['text'] = result['text']
                    
                    # Create tokens dataset with compound dtype
                    dt = np.dtype([
                        ('token', h5py.special_dtype(vlen=str)),
                        ('token_id', np.int32),
                        ('start_timestamp', np.float64),
                        ('end_timestamp', np.float64),
                        ('special_token', np.bool_)
                    ])
                    
                    tokens_ds = group.create_dataset('tokens', (len(result['tokens']),), dtype=dt)
                    
                    # Fill dataset
                    for i, token in enumerate(result['tokens']):
                        tokens_ds[i] = (
                            token['token'],
                            token['token_id'],
                            token['start_timestamp'],
                            token['end_timestamp'],
                            token['special_token']
                        )
            
            # Calculate total stats
            total_token_count = sum(len(r['tokens']) for r in tokenization_results.values())
            char_groups = {k: v for k, v in tokenization_results.items() if k in ['L', 'W']}
            word_groups = {k: v for k, v in tokenization_results.items() if k in ['R', 'S']}
            
            total_char_count = sum(len(r['text']) for r in char_groups.values())
            total_word_count = sum(len(r['text'].split()) for r in word_groups.values())
            
            processing_details['total_token_count'] = total_token_count
            processing_details['total_char_count'] = total_char_count
            processing_details['total_word_count'] = total_word_count
            
            return {
                "local_path": output_path,
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
            test=args.test,
            keep_local=args.keep_local
        )


# Entry point for running the transform from the command line
if __name__ == "__main__":
    LangTransform.run_from_command_line()