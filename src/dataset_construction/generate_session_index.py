import os
import json

def create_session_index_from_flags(flags_filepath="all_flags.json", index_filename="all_sessions.json"):
    """
    Reads the keys (JSON file paths) from an existing flag index file 
    (like all_flags.json) and creates a new index mapping derived dialog names 
    to their full paths.

    Args:
        flags_filepath (str): The path to the JSON file containing flags (e.g., all_flags.json).
        index_filename (str): The name of the JSON file to create with the new index.
    """
    session_map = {}

    try:
        with open(flags_filepath, 'r', encoding='utf-8') as f:
            flags_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Flags file '{flags_filepath}' not found. Cannot generate session index.")
        # Create an empty index file
        try:
             with open(index_filename, 'w', encoding='utf-8') as f:
                 json.dump({}, f, indent=2)
             print(f"Created an empty index file: '{index_filename}'")
        except IOError as e_write:
             print(f"Error: Could not write empty index file '{index_filename}': {e_write}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{flags_filepath}'.")
        return
    except Exception as e:
        print(f"Error reading file '{flags_filepath}': {e}")
        return

    if not isinstance(flags_data, dict):
        print(f"Error: Expected '{flags_filepath}' to contain a JSON object (dictionary).")
        return
        
    paths_processed = 0
    for file_path in flags_data.keys():
        try:
            # Extract filename without extension
            base_name = os.path.basename(file_path)
            dialog_name, _ = os.path.splitext(base_name)
            
            # Use normalized path for consistency
            normalized_path = os.path.normpath(file_path)
            
            if dialog_name in session_map:
                 print(f"Warning: Duplicate dialog name '{dialog_name}' detected. Path '{normalized_path}' will overwrite previous entry for '{session_map[dialog_name]}'.")
            session_map[dialog_name] = normalized_path
            paths_processed += 1
        except Exception as e_path:
            print(f"Warning: Could not process path '{file_path}': {e_path}")

    # Sort the map by dialog name for consistent output
    sorted_session_map = dict(sorted(session_map.items()))

    try:
        with open(index_filename, 'w', encoding='utf-8') as f:
            json.dump(sorted_session_map, f, indent=2, ensure_ascii=False)
        print(f"Successfully created '{index_filename}' with {len(sorted_session_map)} sessions mapped from {paths_processed} paths in '{flags_filepath}'.")
    except IOError as e:
        print(f"Error writing to '{index_filename}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred during writing: {e}")

if __name__ == "__main__":
    create_session_index_from_flags() 