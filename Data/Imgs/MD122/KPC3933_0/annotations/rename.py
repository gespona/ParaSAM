import os
import re
import sys

# Check if the directory path was passed as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python script.py /path/to/png/files")
    sys.exit(1)  # Exit if no argument is given

# The first command-line argument is the script name, so the directory path is the second
directory_path = sys.argv[1]

# Regular expression to match .png files with 'Screenshot (xxxx).png' format
regex_pattern = re.compile(r'^Screenshot \(\d+\)\.png$')

# Filter and sort the list of filenames in the directory
file_list = [f for f in os.listdir(directory_path) if regex_pattern.match(f)]
file_list.sort(key=lambda x: int(re.search(r'\((\d+)\)', x).group(1)))

# Rename files to x.png where x is an integer starting at 1
for i, filename in enumerate(file_list, start=1):
    new_filename = f"{i}.png"
    old_filepath = os.path.join(directory_path, filename)
    new_filepath = os.path.join(directory_path, new_filename)

    # Rename the file
    os.rename(old_filepath, new_filepath)
    print(f'Renamed "{filename}" to "{new_filename}"')

# After running this code, the files in the specified directory will be renamed in order.
