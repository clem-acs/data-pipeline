#!/usr/bin/env python
"""
inspect-lang-window.py - Inspect neural-language query Zarr stores

Usage:
    python inspect-lang-window.py s3://your-bucket/path/to/neuro_lang.zarr
    python inspect-lang-window.py s3://your-bucket/path/to/sessions/your_session_id.zarr
    python inspect-lang-window.py /local/path/to/your/zarr

This script prints detailed information about tokens, windows, and their mappings
from a neural-language query result with the new vectorized storage format.
"""

import sys
import os
import argparse
import zarr
import numpy as np
from pprint import pprint

def decode_bytes(b):
    """Decode bytes to string, if possible."""
    if isinstance(b, bytes):
        try:
            return b.decode('utf-8')
        except UnicodeDecodeError:
            return str(b)
    return b

def inspect_store_metadata(root):
    """Examine store metadata."""
    print("\n=== STORE METADATA ===")
    if hasattr(root, 'attrs') and root.attrs:
        print("Root attributes:")
        for key, value in root.attrs.items():
            print(f"  {key}: {value}")
    else:
        print("No store metadata found.")

def inspect_zarr(uri):
    """Open and inspect a Zarr store."""
    print(f"Opening Zarr store: {uri}")
    
    # Determine storage options based on URI
    if uri.startswith('s3://'):
        storage_options = {"anon": False}
    else:
        storage_options = {}
    
    # Open store
    try:
        root = zarr.open_group(store=uri, mode="r", storage_options=storage_options)
    except Exception as e:
        print(f"Error opening store: {e}")
        return
    
    # Handle the case of empty store
    if not list(root.keys()):
        print("WARNING: Store appears to be empty or not fully initialized.")
        inspect_store_metadata(root)
        return
    
    # Check if this is a session or the main query store
    session = None
    if "sessions" in root:
        # This is the main query store
        print("This is a query store containing multiple sessions.")
        session_keys = list(root['sessions'].keys())
        print(f"Available sessions: {session_keys}")
        
        if not session_keys:
            print("No sessions available in the store.")
            inspect_store_metadata(root)
            return
        
        # Use the first session
        session_id = session_keys[0]
        print(f"Examining the first session: {session_id}")
        session = root['sessions'][session_id]
    else:
        # Check if this might be a single session directly
        if any(key in root for key in ["tokens", "windows", "token_window_map"]):
            print("This appears to be a single session store.")
            session = root
        else:
            print("This doesn't appear to be a neuro_lang query store or session. Available keys:")
            print(list(root.keys()))
            inspect_store_metadata(root)
            return
    
    # Check for the expected structure
    print("\n=== STORE STRUCTURE ===")
    session_keys = list(session.keys())
    print(f"Session keys: {session_keys}")
    
    # Tokens
    if "tokens" in session:
        tokens = session["tokens"]
        print("\n=== TOKENS ===")
        token_keys = list(tokens.keys())
        print(f"Token array keys: {token_keys}")
        
        # Print shapes
        print("\nToken array shapes:")
        for key in token_keys:
            try:
                array = tokens[key]
                print(f"  {key}: {array.shape}, {array.dtype}")
            except Exception as e:
                print(f"  {key}: Error accessing array ({str(e)})")
        
        # Print first 10 tokens
        token_count = 0
        try:
            token_count = tokens["text"].shape[0] if "text" in tokens else 0
            print(f"\nTotal tokens: {token_count}")
            
            if token_count > 0:
                print("\nFirst 10 tokens:")
                for i in range(min(10, token_count)):
                    token_data = {}
                    for key in token_keys:
                        try:
                            value = tokens[key][i]
                            if isinstance(value, np.ndarray) and value.dtype.kind in ('S', 'a'):
                                value = decode_bytes(value)
                            token_data[key] = value
                        except Exception as e:
                            token_data[key] = f"<Error: {str(e)}>"
                    
                    print(f"Token {i}:")
                    pprint(token_data)
        except Exception as e:
            print(f"Error processing tokens: {str(e)}")
    
    # Windows
    if "windows" in session:
        windows = session["windows"]
        print("\n=== WINDOWS ===")
        window_keys = list(windows.keys())
        print(f"Window array keys: {window_keys}")
        
        # Print shapes
        print("\nWindow array shapes:")
        for key in window_keys:
            try:
                array = windows[key]
                print(f"  {key}: {array.shape}, {array.dtype}")
            except Exception as e:
                print(f"  {key}: Error accessing array ({str(e)})")
        
        # Calculate window count
        window_count = 0
        try:
            if window_keys:
                for key in window_keys:
                    try:
                        array = windows[key]
                        window_count = array.shape[0]
                        break
                    except:
                        continue
            
            print(f"\nTotal windows: {window_count}")
            
            # Print first window time info
            if window_count > 0 and "time" in windows:
                print("\nFirst 5 window times:")
                for i in range(min(5, window_count)):
                    print(f"  Window {i}: {windows['time'][i]}")
        except Exception as e:
            print(f"Error processing windows: {str(e)}")
    
    # Token-Window Mapping
    if "token_window_map" in session:
        try:
            mapping = session["token_window_map"]
            print("\n=== TOKEN-WINDOW MAPPING ===")
            mapping_keys = list(mapping.keys())
            print(f"Mapping keys: {mapping_keys}")
            
            # Print shapes
            print("\nMapping array shapes:")
            for key in mapping_keys:
                try:
                    array = mapping[key]
                    print(f"  {key}: {array.shape}, {array.dtype}")
                except Exception as e:
                    print(f"  {key}: Error accessing array ({str(e)})")
            
            # Calculate mapping count
            mapping_count = 0
            if "token_idx" in mapping:
                mapping_count = mapping["token_idx"].shape[0]
            
            print(f"\nTotal mappings: {mapping_count}")
            
            # Print statistics about windows per token
            if mapping_count > 0 and "tokens" in session:
                token_indices = mapping["token_idx"][:]
                unique_tokens = np.unique(token_indices)
                token_count = session["tokens"]["text"].shape[0] if "text" in session["tokens"] else 0
                
                print(f"\nUnique tokens with windows: {len(unique_tokens)} out of {token_count}")
                
                # Count windows per token
                windows_per_token = {}
                for token_idx in token_indices:
                    windows_per_token[token_idx] = windows_per_token.get(token_idx, 0) + 1
                
                if windows_per_token:
                    window_counts = list(windows_per_token.values())
                    print(f"Windows per token: min={min(window_counts)}, max={max(window_counts)}, avg={sum(window_counts)/len(window_counts):.1f}")
                
                # Show a few sample mappings with token text
                if mapping_count > 0 and "token_idx" in mapping and "window_idx" in mapping:
                    print("\nFirst 10 token-window mappings with context:")
                    for i in range(min(10, mapping_count)):
                        token_idx = mapping["token_idx"][i]
                        window_idx = mapping["window_idx"][i]
                        
                        # Get token text and info
                        token_text = "N/A"
                        if "text" in session["tokens"]:
                            try:
                                token_text = decode_bytes(session["tokens"]["text"][token_idx])
                            except:
                                token_text = "<error>"
                        
                        token_info = f"Token {token_idx} (\"{token_text}\")"
                        
                        # Get window time
                        window_time = "N/A"
                        if "time" in windows:
                            try:
                                window_time = windows["time"][window_idx]
                            except:
                                window_time = "<error>"
                        
                        print(f"  Mapping {i}: {token_info} → Window {window_idx} (t={window_time})")
        except Exception as e:
            print(f"Error processing token-window mapping: {str(e)}")
    
    # Token Sequences (old format compatibility)
    if "token_sequence" in session:
        try:
            print("\n=== TOKEN SEQUENCE ===")
            sequence = session["token_sequence"]
            if isinstance(sequence, zarr.Group):
                sequence_keys = list(sequence.keys())
                sequence_length = len(sequence_keys)
                print(f"Token sequence length: {sequence_length}")
                
                # Print first 10 sequence indices
                if sequence_length > 0:
                    print("\nFirst 10 sequence indices:")
                    for i in range(min(10, sequence_length)):
                        if str(i) in sequence.attrs:
                            print(f"  Pos {i}: {sequence.attrs[str(i)]}")
        except Exception as e:
            print(f"Error processing token sequence: {str(e)}")
    
    # Metadata
    print("\n=== METADATA ===")
    if hasattr(session, 'attrs'):
        print("Session attributes:")
        for key, value in session.attrs.items():
            print(f"  {key}: {value}")
    else:
        print("No session attributes found.")

def main():
    parser = argparse.ArgumentParser(description="Inspect a neural-language query Zarr store")
    parser.add_argument("uri", help="URI of the Zarr store to inspect (S3 URI or local path)")
    args = parser.parse_args()
    
    inspect_zarr(args.uri)

if __name__ == "__main__":
    main()