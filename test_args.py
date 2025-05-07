#!/usr/bin/env python
import argparse

def main():
    parser = argparse.ArgumentParser(description='Test how argparse handles hyphens in argument names')
    parser.add_argument('--keep-local', action='store_true', help='Test flag with hyphen')
    parser.add_argument('--test', action='store_true', help='Test flag')
    
    args = parser.parse_args()
    
    print("args.keep_local:", args.keep_local)
    print("hasattr(args, 'keep_local'):", hasattr(args, 'keep_local'))
    print("hasattr(args, 'keep-local'):", hasattr(args, 'keep-local'))

if __name__ == "__main__":
    main()