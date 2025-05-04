"""
Language tokenization transform for curated H5 files.

This transform:
1. Reads curated H5 files from S3
2. Extracts language/L (listen) and language/W (written) character data
3. Tokenizes the text using a specified tokenizer
4. Saves results to processed/lang/ prefix in S3
5. Records metadata in DynamoDB
"""

import os
import h5py
import numpy as np
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer

try:
    # When running as an installed package
    from ..base import DataTransform
except ImportError:
    # When running as a script
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from base import DataTransform


class LangTransform(DataTransform):
    """Language tokenization transform for curated H5 files.
    
    This transform tokenizes language data from curated H5 files,
    using the specified tokenizer and saves the results to S3.
    """
    
    def __init__(self, source_prefix: str = 'curated-h5/',
                 dest_prefix: str = 'processed/lang/',
                 tokenizer_name: str = 'gpt2',
                 **kwargs):
        """Initialize the language tokenization transform.
        
        Args:
            source_prefix: S3 prefix for source data (curated H5 files)
            dest_prefix: S3 prefix for destination data (processed files)
            tokenizer_name: Name of the tokenizer to use ('gpt2', 'facebook/opt-125m', etc.)
            **kwargs: Additional arguments for DataTransform
        """
        # Set default transform info
        transform_id = kwargs.pop('transform_id', 'lang_v0')
        script_id = kwargs.pop('script_id', '0L')
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
        self.source_prefix = source_prefix
        self.dest_prefix = dest_prefix
        self.tokenizer_name = tokenizer_name
        self._tokenizer = None
        
        self.logger.info(f"Language tokenization transform initialized with:")
        self.logger.info(f"  Source prefix: {self.source_prefix}")
        self.logger.info(f"  Destination prefix: {self.dest_prefix}")
        self.logger.info(f"  Tokenizer: {self.tokenizer_name}")
    
    @property
    def tokenizer(self):
        """Lazy-load tokenizer when needed."""
        if self._tokenizer is None:
            self.logger.info(f"Loading tokenizer: {self.tokenizer_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        return self._tokenizer
    
    def find_items_to_process(self) -> List[str]:
        """Find H5 files that need to be processed.
        
        Returns:
            List of file IDs to process
        
        Note: The DataTransform base class handles deduplication by filtering out
        files that have already been processed.
        """
        self.logger.info(f"Listing H5 files in {self.source_prefix}")
        
        # List all H5 files in the source prefix (curated-h5/)
        h5_files = []
        paginator = self.s3.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=self.source_prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].endswith('.h5'):
                        file_name = os.path.basename(obj['Key'])
                        file_id = os.path.splitext(file_name)[0]
                        h5_files.append(file_id)
        
        self.logger.info(f"Found {len(h5_files)} H5 files")
        return h5_files
    
    def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get file information for a file ID.
        
        Args:
            file_id: ID of the file to look up
            
        Returns:
            Dict with file information or None if not found
        """
        for page in self.s3.get_paginator('list_objects_v2').paginate(
                Bucket=self.s3_bucket, Prefix=self.source_prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].endswith('.h5'):
                        current_file_id = os.path.splitext(os.path.basename(obj['Key']))[0]
                        if current_file_id == file_id:
                            return {
                                'file_id': file_id,
                                'file_key': obj['Key'],
                                'file_size': obj['Size']
                            }
        return None
    
    def analyze_file(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a file to determine if it has language data to process.
        
        Args:
            file_info: Dict with file information
            
        Returns:
            Dict with analysis results including qualification
        """
        try:
            temp_path = self.create_temp_path(os.path.basename(file_info['file_key']))
            self.download_s3_file(file_info['file_key'], temp_path)
            
            # Initialize analysis results
            analysis = {
                'file_id': file_info['file_id'],
                'file_key': file_info['file_key'],
                'has_L_data': False,
                'has_W_data': False,
                'L_chars_count': 0,
                'W_chars_count': 0
            }
            
            with h5py.File(temp_path, 'r') as f:
                # Check if file has language group
                if 'language' not in f:
                    self.logger.info(f"No 'language' group found in {file_info['file_id']}")
                    self.cleanup_temp_file(temp_path)
                    return {'qualifies': False, **analysis}
                
                # Check for L and W subgroups with chars datasets
                if 'L' in f['language'] and 'chars' in f['language']['L']:
                    analysis['has_L_data'] = True
                    analysis['L_chars_count'] = len(f['language/L/chars'])
                    self.logger.info(f"Found 'L' data with {analysis['L_chars_count']} characters")
                
                if 'W' in f['language'] and 'chars' in f['language']['W']:
                    analysis['has_W_data'] = True
                    analysis['W_chars_count'] = len(f['language/W/chars'])
                    self.logger.info(f"Found 'W' data with {analysis['W_chars_count']} characters")
            
            # Cleanup
            self.cleanup_temp_file(temp_path)
            
            # The file qualifies if it has either L or W data
            qualifies = analysis['has_L_data'] or analysis['has_W_data']
            
            analysis['qualifies'] = qualifies
            
            if qualifies:
                self.logger.info(f"File {file_info['file_id']} has language data to tokenize")
            else:
                self.logger.info(f"File {file_info['file_id']} doesn't have language data")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing file {file_info['file_id']}: {e}")
            
            # Clean up the temporary file if it exists
            if 'temp_path' in locals():
                self.cleanup_temp_file(temp_path)
            
            return {
                'file_id': file_info['file_id'],
                'file_key': file_info['file_key'],
                'qualifies': False,
                'error': str(e)
            }
    
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
    
    def process_language(self, file_info: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Process language data and create tokenized output.
        
        Args:
            file_info: Dict with file information
            analysis: Analysis results from analyze_file
            
        Returns:
            Dict with transform record
        """
        file_id = file_info['file_id']
        file_key = file_info['file_key']
        
        # Create destination key (keeping same filename)
        file_name = os.path.basename(file_key)
        dest_key = f"{self.dest_prefix}{file_name}"
        
        # Create source and destination paths for recording
        source_path = f"s3://{self.s3_bucket}/{file_key}"
        dest_path = f"s3://{self.s3_bucket}/{dest_key}"
        
        self.logger.info(f"Processing file {file_id}")
        
        try:
            # Download the file
            temp_path = self.create_temp_path(file_name)
            processed_path = f"{temp_path}.processed"
            self.download_s3_file(file_key, temp_path)
            
            # Process the file
            tokenization_results = {}
            
            with h5py.File(temp_path, 'r') as f_in:
                # Process both L and W groups if they exist
                if analysis['has_L_data']:
                    result = self.extract_and_tokenize(f_in, 'L')
                    if result:
                        tokenization_results['L'] = result
                        self.logger.info(f"Tokenized {len(result['text'])} chars into {len(result['tokens'])} tokens for L group")
                
                if analysis['has_W_data']:
                    result = self.extract_and_tokenize(f_in, 'W')
                    if result:
                        tokenization_results['W'] = result
                        self.logger.info(f"Tokenized {len(result['text'])} chars into {len(result['tokens'])} tokens for W group")
            
            # If nothing was processed, skip
            if not tokenization_results:
                self.logger.warning(f"No language data processed for {file_id}")
                self.cleanup_temp_file(temp_path)
                return {"status": "skipped"}
            
            # Create output file
            with h5py.File(processed_path, 'w') as f_out:
                lang_group = f_out.create_group('language')
                lang_group.attrs['tokenizer'] = self.tokenizer_name
                
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
            
            # Upload to S3
            if not self.dry_run:
                self.upload_s3_file(processed_path, dest_key)
                self.logger.info(f"Uploaded tokenized file to {dest_path}")
            else:
                self.logger.info(f"[DRY RUN] Would upload tokenized file to {dest_path}")
            
            # Cleanup temp files
            self.cleanup_temp_file(temp_path)
            self.cleanup_temp_file(processed_path)
            
            # Calculate stats for metadata
            token_count = sum(len(r['tokens']) for r in tokenization_results.values())
            char_count = sum(len(r['text']) for r in tokenization_results.values())
            
            # Record successful transform
            return self.record_transform(
                data_id=file_id,
                transform_metadata={
                    'processing_type': 'language_tokenization',
                    'tokenizer': self.tokenizer_name,
                    'token_count': token_count,
                    'char_count': char_count,
                    'L_chars_count': analysis.get('L_chars_count', 0),
                    'W_chars_count': analysis.get('W_chars_count', 0)
                },
                source_paths=[source_path],
                destination_paths=[dest_path],
                status='success',
                parent_transforms=['curation_v0']  # Add parent transform for lineage tracking
            )
        
        except Exception as e:
            self.logger.error(f"Error processing file {file_id}: {e}")
            
            # Clean up temp files
            if 'temp_path' in locals():
                self.cleanup_temp_file(temp_path)
            if 'processed_path' in locals():
                self.cleanup_temp_file(processed_path)
            
            # Record failed transform
            return self.record_transform(
                data_id=file_id,
                transform_metadata={
                    'processing_type': 'language_tokenization',
                    'tokenizer': self.tokenizer_name,
                    'error': str(e)
                },
                source_paths=[source_path],
                status='failed',
                error_details=str(e),
                parent_transforms=['curation_v0']  # Add parent transform for lineage tracking
            )
    
    def process_item(self, file_id: str) -> Dict[str, Any]:
        """Process a single file.
        
        Args:
            file_id: ID of the file to process
            
        Returns:
            Dict with processing result
        """
        try:
            # Get file information
            file_info = self.get_file_info(file_id)
            if not file_info:
                self.logger.warning(f"File with ID {file_id} not found")
                return {"status": "skipped"}
            
            # Analyze the file to check if it has language data
            analysis = self.analyze_file(file_info)
            
            # Process if it has any language data
            if analysis.get('qualifies', False):
                return self.process_language(file_info, analysis)
            else:
                self.logger.info(f"File {file_id} doesn't have language data")
                return {"status": "skipped"}
        
        except Exception as e:
            self.logger.error(f"Error processing file {file_id}: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    @classmethod
    def add_subclass_arguments(cls, parser):
        """Add language tokenization-specific command-line arguments.
        
        Args:
            parser: ArgumentParser instance
        """
        parser.add_argument('--source-prefix', type=str, default='curated-h5/',
                          help='S3 prefix for source data')
        parser.add_argument('--dest-prefix', type=str, default='processed/lang/',
                          help='S3 prefix for destination data')
        parser.add_argument('--tokenizer', type=str, default='gpt2',
                          help='Name of the tokenizer to use (e.g., gpt2, facebook/opt-125m)')
        parser.add_argument('--s3-bucket', type=str, default='conduit-data-dev',
                          help='S3 bucket name')
    
    @classmethod
    def from_args(cls, args):
        """Create a transform instance from command-line arguments.
        
        Args:
            args: Parsed command-line arguments
            
        Returns:
            Instance of LangTransform
        """
        return cls(
            source_prefix=args.source_prefix,
            dest_prefix=args.dest_prefix,
            tokenizer_name=args.tokenizer,
            s3_bucket=args.s3_bucket,
            verbose=args.verbose,
            log_file=args.log_file,
            dry_run=args.dry_run
        )


# Entry point for running the transform from the command line
if __name__ == "__main__":
    LangTransform.run_from_command_line()