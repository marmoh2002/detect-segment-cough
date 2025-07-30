#!/usr/bin/env python3
"""
Program to segment coughs from audio files located in a nested
directory structure. It scans a root folder containing date-based
subfolders, which in turn contain participant folders.
(e.g., root_folder/date1/name1/, root_folder/date2/name4/)
"""
# bagus@ep.its.ac.id
# Modified by a helpful chat assistant

import librosa
import os
import sys
import argparse
import soundfile as sf

# Assuming your custom modules are in the specified paths
sys.path.append('./src/')
sys.path.append(os.path.abspath('./src/cough_detection/'))
from src.segmentation import segment_cough

def process_file(input_file, dir_output, fs_out=16000):
    """
    Processes a single audio file to find and save cough segments.
    (This function remains unchanged)
    """
    print(f"Segmenting cough from {input_file} -> to -> {dir_output}")
    try:
        x, fs = librosa.load(input_file, sr=fs_out)
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        return

    cough_segments, cough_mask = segment_cough(x, fs, cough_padding=0)
    
    if not cough_segments:
        print(f"No cough segments found in {input_file}.")
        return

    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
        print(f"Created directory: {dir_output}")

    print(f"Found {len(cough_segments)} cough segments in {os.path.basename(input_file)}")
    
    for i, segment in enumerate(cough_segments):
        output_filename = f"cough-{i+1}.wav"
        output_path = os.path.join(dir_output, output_filename)
        
        sf.write(output_path, segment, fs)
        print(f"  > Wrote segment to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segment coughs from a nested date/participant directory structure.')
    
    parser.add_argument('-i', '--input_dir', required=True, type=str,
                        help='Root directory containing date folders (e.g., ./folder).')
                        
    parser.add_argument('-o', '--output_dir', default='./segmented_coughs', type=str,
                        help='Base output directory to store the results (e.g., ./newdir).')

    parser.add_argument('-f', '--filename', default='cough-heavy.wav', type=str,
                        help='The name of the audio file to find within each participant folder.')
                        
    parser.add_argument('-fs', '--fs_out', default=16000, type=int,
                        help='Output sampling rate.')
    
    args = parser.parse_args()

    # --- NEW MAIN LOOP ---
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)

    print(f"Starting scan in root directory: {args.input_dir}")
    print(f"Looking for files named '{args.filename}' in participant subfolders.")
    print("-" * 50)

    # Level 1: Iterate through items in the root directory (the 'date' folders)
    for date_folder_name in sorted(os.listdir(args.input_dir)):
        date_folder_path = os.path.join(args.input_dir, date_folder_name)

        # Ensure the item is a directory before proceeding
        if os.path.isdir(date_folder_path):
            # Level 2: Iterate through items in the date folder (the 'name' folders)
            for participant_name in sorted(os.listdir(date_folder_path)):
                participant_folder_path = os.path.join(date_folder_path, participant_name)

                # Ensure this is also a directory
                if os.path.isdir(participant_folder_path):
                    # Construct the full path to the source audio file
                    input_file_path = os.path.join(participant_folder_path, args.filename)

                    if os.path.exists(input_file_path):
                        print(f"Found participant '{participant_name}' in '{date_folder_name}'")

                        # --- Construct the output path ---
                        # The key change is here: We use participant_name directly under the
                        # main output_dir, ignoring the date_folder_name.
                        specific_output_dir = os.path.join(args.output_dir, participant_name)
                        
                        # Call the processing function
                        process_file(input_file_path, specific_output_dir, args.fs_out)
                        print("-" * 50)
                    else:
                        print(f"Skipping '{participant_folder_path}': File '{args.filename}' not found.")
                        print("-" * 50)

    print("All processing complete.")