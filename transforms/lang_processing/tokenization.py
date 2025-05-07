"""
Tokenization functions for processing text data from H5 files.
"""

import json

def extract_and_tokenize(h5_file, group_name, tokenizer):
    """Extract and tokenize text from a character-level language group.
    
    Args:
        h5_file: Open h5py.File object
        group_name: 'L' or 'W' group name
        tokenizer: HuggingFace tokenizer instance
        
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
    encoding = tokenizer(text, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    token_ids = encoding['input_ids']
    offsets = encoding['offset_mapping']
    
    # Map tokens to characters
    token_data = []
    
    for token_idx, (start_offset, end_offset) in enumerate(offsets):
        # Handle special tokens
        if start_offset == end_offset:
            special_token_data = {
                'token': tokens[token_idx],
                'token_id': token_ids[token_idx],
                'start_timestamp': 0,
                'end_timestamp': 0,
                'special_token': True
            }
            
            # Add empty keystroke fields for W group
            if group_name == 'W':
                special_token_data.update({
                    'keystroke_events': json.dumps([]),  # Empty JSON array
                    'trigger_keystrokes': json.dumps([]),  # Empty JSON array
                    'char_indices': []
                })
                
            token_data.append(special_token_data)
            continue
        
        # Find characters for this token
        chars_in_token = []
        char_indices = []
        all_keystrokes = []
        all_trigger_keystrokes = []
        
        for i in range(len(text)):
            if i >= start_offset and i < end_offset:
                char_idx = i
                if char_idx < len(chars_data):
                    char_data = chars_data[char_idx]
                    chars_in_token.append(char_data)
                    char_indices.append(char_idx)
                    
                    # Only collect keystroke data for W group
                    if group_name == 'W':
                        # Collect keystroke data if available and serialize to JSON
                        if 'keystrokes' in char_data and char_data['keystrokes'] is not None:
                            # Serialize each keystroke object to JSON
                            for keystroke in char_data['keystrokes']:
                                if keystroke is not None:
                                    try:
                                        all_keystrokes.append(json.dumps(keystroke))
                                    except (TypeError, ValueError):
                                        # Handle case where keystroke can't be serialized
                                        all_keystrokes.append(json.dumps(str(keystroke)))
                        
                        if 'trigger_keystroke' in char_data and char_data['trigger_keystroke'] is not None:
                            try:
                                all_trigger_keystrokes.append(json.dumps(char_data['trigger_keystroke']))
                            except (TypeError, ValueError):
                                # Handle case where trigger_keystroke can't be serialized
                                all_trigger_keystrokes.append(json.dumps(str(char_data['trigger_keystroke'])))
        
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
            
            # Base token data
            token_entry = {
                'token': tokens[token_idx],
                'token_id': token_ids[token_idx],
                'start_timestamp': start_timestamp,
                'end_timestamp': end_timestamp,
                'special_token': False
            }
            
            # Add keystroke data for W group only
            if group_name == 'W':
                # Keystroke events are already serialized as JSON strings at this point
                token_entry.update({
                    'keystroke_events': json.dumps(all_keystrokes),  # Wrap in JSON array
                    'trigger_keystrokes': json.dumps(all_trigger_keystrokes),  # Wrap in JSON array
                    'char_indices': char_indices
                })
                
            token_data.append(token_entry)
        else:
            # Should rarely happen, but handle just in case
            token_entry = {
                'token': tokens[token_idx],
                'token_id': token_ids[token_idx],
                'start_timestamp': 0,
                'end_timestamp': 0,
                'special_token': False
            }
            
            # Add empty keystroke data for W group
            if group_name == 'W':
                token_entry.update({
                    'keystroke_events': json.dumps([]),  # Empty JSON array
                    'trigger_keystrokes': json.dumps([]),  # Empty JSON array
                    'char_indices': []
                })
                
            token_data.append(token_entry)
    
    return {'text': text, 'tokens': token_data}

def extract_and_tokenize_words(h5_file, group_name, tokenizer):
    """Extract and tokenize text from a word-level language group (R or S).
    
    Args:
        h5_file: Open h5py.File object
        group_name: 'R' or 'S' group name
        tokenizer: HuggingFace tokenizer instance
        
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
    encoding = tokenizer(text, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
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
            token_data.append({
                'token': tokens[token_idx],
                'token_id': token_ids[token_idx],
                'start_timestamp': 0,
                'end_timestamp': 0,
                'special_token': False
            })
    
    return {'text': text, 'tokens': token_data}
