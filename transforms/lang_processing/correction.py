"""
Functions for spelling correction and token alignment.
"""

import re
import difflib
from typing import Dict, List, Tuple, Any, Optional
from autocorrect import Speller

# Initialize once for English
EN_SPELLER = Speller(lang='en')

def correct_text(text: str) -> str:
    """Correct spelling in the given text using Autocorrect.
    
    Args:
        text: Text to correct
        
    Returns:
        Corrected text
    """
    # Split text into words and non-word segments
    # This preserves spacing and punctuation
    segments = re.findall(r'[\w\']+|[^\w\']+', text)
    
    corrected_segments = []
    for segment in segments:
        # Only correct word-like segments
        if re.match(r'^[\w\']+$', segment):
            # Skip very short words or likely deliberate non-words (acronyms)
            if len(segment) <= 1 or segment.isupper():
                corrected_segments.append(segment)
                continue
                
            # Correct the word using pre-initialized speller
            corrected = EN_SPELLER(segment)
            
            # Preserve original capitalization if changed
            if segment[0].isupper() and not corrected[0].isupper():
                corrected = corrected.capitalize()
            
            corrected_segments.append(corrected)
        else:
            # Keep non-word segments (spaces, punctuation) as is
            corrected_segments.append(segment)
    
    return ''.join(corrected_segments)

def align_and_timestamp(original_tokens: List[Dict], corrected_text: str, tokenizer: Any) -> Tuple[List[Dict], float]:
    """Align original and corrected tokens and assign timestamps.
    
    Args:
        original_tokens: List of original tokens with timestamps
        corrected_text: Corrected text string
        tokenizer: HuggingFace tokenizer to use
        
    Returns:
        Tuple of (corrected tokens with timestamps, correctness score)
    """
    # Generate corrected tokens
    corrected_encoding = tokenizer(corrected_text, return_offsets_mapping=True)
    corrected_token_strs = tokenizer.convert_ids_to_tokens(corrected_encoding['input_ids'])
    corrected_token_ids = corrected_encoding['input_ids']
    
    # Create basic structure for corrected tokens
    corrected_tokens = []
    for i, token in enumerate(corrected_token_strs):
        corrected_tokens.append({
            'token': token,
            'token_id': corrected_token_ids[i],
            'special_token': corrected_encoding['offset_mapping'][i][0] == corrected_encoding['offset_mapping'][i][1],
            # Timestamps will be filled later
            'start_timestamp': 0,
            'end_timestamp': 0,
            'original_indices': [],
            'unchanged': False
        })
    
    # Extract just the token text for alignment
    original_token_strs = [t['token'] for t in original_tokens]
    corrected_token_strs = [t['token'] for t in corrected_tokens]
    
    # Use difflib to align tokens
    matcher = difflib.SequenceMatcher(None, original_token_strs, corrected_token_strs)
    
    unchanged_count = 0
    total_mapped = 0
    
    # Process each alignment operation
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Tokens are identical - direct mapping of timestamps
            for k, orig_idx in enumerate(range(i1, i2)):
                corr_idx = j1 + k
                original_token = original_tokens[orig_idx]
                
                corrected_tokens[corr_idx]['start_timestamp'] = original_token['start_timestamp']
                corrected_tokens[corr_idx]['end_timestamp'] = original_token['end_timestamp']
                corrected_tokens[corr_idx]['original_indices'] = [orig_idx]
                corrected_tokens[corr_idx]['unchanged'] = True
                
                unchanged_count += 1
                total_mapped += 1
                
        elif tag == 'replace':
            # Map multiple original tokens to multiple corrected tokens
            orig_range = list(range(i1, i2))
            corr_range = list(range(j1, j2))
            
            if orig_range and corr_range:
                # Get all timestamps from original range
                start_times = [original_tokens[idx]['start_timestamp'] for idx in orig_range]
                end_times = [original_tokens[idx]['end_timestamp'] for idx in orig_range]
                
                # For each corrected token, assign timestamps proportionally
                segment_count = len(corr_range)
                for k, corr_idx in enumerate(corr_range):
                    # Use min start and max end for the whole range
                    corrected_tokens[corr_idx]['start_timestamp'] = min(start_times) if start_times else 0
                    corrected_tokens[corr_idx]['end_timestamp'] = max(end_times) if end_times else 0
                    corrected_tokens[corr_idx]['original_indices'] = orig_range
                    corrected_tokens[corr_idx]['unchanged'] = False
                    
                    total_mapped += 1
        
        elif tag == 'insert':
            # Tokens in corrected but not in original - use nearest original for timestamps
            for corr_idx in range(j1, j2):
                # Find nearest valid original token
                if i1 > 0:
                    # Use preceding token
                    nearest_idx = i1 - 1
                elif i2 < len(original_tokens):
                    # Use following token
                    nearest_idx = i2
                else:
                    # No reference, use zeros
                    corrected_tokens[corr_idx]['start_timestamp'] = 0
                    corrected_tokens[corr_idx]['end_timestamp'] = 0
                    corrected_tokens[corr_idx]['original_indices'] = []
                    continue
                
                # Use nearest token's timestamps
                corrected_tokens[corr_idx]['start_timestamp'] = original_tokens[nearest_idx]['start_timestamp']
                corrected_tokens[corr_idx]['end_timestamp'] = original_tokens[nearest_idx]['end_timestamp']
                corrected_tokens[corr_idx]['original_indices'] = [nearest_idx]
        
        # 'delete' operations (original tokens not in corrected) are handled implicitly
    
    # Calculate correctness score (percentage of unchanged tokens)
    correctness_score = unchanged_count / total_mapped if total_mapped > 0 else 0.0
    
    return corrected_tokens, correctness_score

def correct_and_align_w_group(chars_data: List[Dict], tokenizer: Any) -> Tuple[Dict, float]:
    """Process 'W' group data for spelling correction, tokenization, and alignment.
    
    Args:
        chars_data: List of character data from the W group
        tokenizer: HuggingFace tokenizer instance
        
    Returns:
        Tuple of (result dict with tokens and text, correctness score)
    """
    # Sort characters by timestamp
    sorted_chars = sorted(chars_data, key=lambda x: x['timestamp'])
    
    # Regenerate original text
    original_text = ''.join(c['char'] for c in sorted_chars)
    
    # Perform spelling correction
    corrected_text = correct_text(original_text)
    
    # Tokenize original text and get timestamps
    # This is similar to the existing extract_and_tokenize function
    original_encoding = tokenizer(original_text, return_offsets_mapping=True)
    original_tokens = tokenizer.convert_ids_to_tokens(original_encoding['input_ids'])
    original_token_ids = original_encoding['input_ids']
    original_offsets = original_encoding['offset_mapping']
    
    # Map timestamps to original tokens (similar to extract_and_tokenize)
    original_tokens_with_timestamps = []
    
    for token_idx, (start_offset, end_offset) in enumerate(original_offsets):
        # Handle special tokens
        if start_offset == end_offset:
            original_tokens_with_timestamps.append({
                'token': original_tokens[token_idx],
                'token_id': original_token_ids[token_idx],
                'start_timestamp': 0,
                'end_timestamp': 0,
                'special_token': True
            })
            continue
        
        # Find characters for this token using exact same approach as existing code
        chars_in_token = []
        for i in range(len(original_text)):
            if i >= start_offset and i < end_offset:
                char_idx = i
                if char_idx < len(sorted_chars):
                    chars_in_token.append(sorted_chars[char_idx])
        
        if chars_in_token:
            # Use first char's timestamp to last char's timestamp
            start_timestamp = chars_in_token[0]['timestamp']
            end_timestamp = chars_in_token[-1]['timestamp']
            
            original_tokens_with_timestamps.append({
                'token': original_tokens[token_idx],
                'token_id': original_token_ids[token_idx],
                'start_timestamp': start_timestamp,
                'end_timestamp': end_timestamp,
                'special_token': False
            })
        else:
            # Fallback if no chars found
            original_tokens_with_timestamps.append({
                'token': original_tokens[token_idx],
                'token_id': original_token_ids[token_idx],
                'start_timestamp': 0,
                'end_timestamp': 0,
                'special_token': False
            })
    
    # Align and get corrected tokens with timestamps
    corrected_tokens, correctness_score = align_and_timestamp(
        original_tokens_with_timestamps, 
        corrected_text, 
        tokenizer
    )
    
    # Return result in the same format as extract_and_tokenize
    return {
        'text': corrected_text,
        'tokens': corrected_tokens,
        'correctness_score': correctness_score
    }, correctness_score