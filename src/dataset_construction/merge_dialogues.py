import os
import json
import re
from collections import defaultdict
import copy

def extract_parts_from_filename(filename):
    """
    Extracts the acronym, scenario, and suffix from a JSON filename.
    Example: FOR_ApothecaryGoblins_AD_Boss.json -> ('FOR', 'ApothecaryGoblins', 'AD_Boss')
    Example: QST_FindMol_AD_TieflingKids.json -> ('QST', 'FindMol', 'AD_TieflingKids')
    Example: DEN_DenOfSevenBones_PAD_MeetingEnverGortash.json -> ('DEN', 'DenOfSevenBones', 'PAD_MeetingEnverGortash')
    
    This function preserves the original case of each part.
    
    Returns None if the pattern doesn't match.
    """    
    # Special case handling for known problematic files with LaeZel vs Laezel
    
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split('_', 2) # Split into max 3 parts: Acronym, Scenario, Suffix

    if len(parts) == 3:
        acronym, scenario, suffix = parts
        if acronym and scenario and suffix: # Ensure all parts are non-empty
             # Handle cases like AD_TieflingKids vs PAD_MeetingEnverGortash
             # The suffix is everything after the second underscore.
            return acronym.lower(), scenario.lower(), suffix
    elif len(parts) == 2:
         acronym, scenario = parts
         if acronym and scenario:
             return acronym.lower(), scenario.lower(), "" # Return empty string for suffix

    # Return None if parsing failed for validation purposes
    return None, None, None

def _collect_all_node_ids_recursively(node_data, all_ids_set):
    """
    Recursively collects all 'id' field values from a node and its children.
    """
    if not isinstance(node_data, dict):
        return
    
    current_id = node_data.get('id')
    if current_id: # Ensure there is an ID and it's not empty/None
        all_ids_set.add(current_id)
        
    children = node_data.get('children')
    if isinstance(children, dict):
        for child_node in children.values():
            _collect_all_node_ids_recursively(child_node, all_ids_set)

def update_node_ids_recursive(node, id_prefix, original_file_node_ids):
    """
    Recursively updates node IDs, goto, link, and children keys with a prefix.
    Only prefixes internal goto/links.
    """
    original_id = node.get('id', '')
    new_id = f"{id_prefix}_{original_id}" if id_prefix else original_id

    # Update the node's own ID
    node['id'] = new_id

    # Update goto if it points to a node within the same original file
    original_goto = node.get('goto', '')
    if original_goto and original_goto in original_file_node_ids:
        node['goto'] = f"{id_prefix}_{original_goto}" if id_prefix else original_goto

    # Update link if it points to a node within the same original file
    original_link = node.get('link', '')
    if original_link and original_link in original_file_node_ids:
         node['link'] = f"{id_prefix}_{original_link}" if id_prefix else original_link # Corrected from node['goto']


    # Recursively update children
    new_children = {}
    if 'children' in node and node['children']:
        for child_id, child_node in node['children'].items():
            # Deep copy child node to avoid modifying the original structure shared across loops
            updated_child_node = copy.deepcopy(child_node)
            # Recursively update the child node
            update_node_ids_recursive(updated_child_node, id_prefix, original_file_node_ids)
            # Use the child's *new* ID as the key in the parent's new children dict
            new_children[updated_child_node['id']] = updated_child_node
    node['children'] = new_children

def merge_json_files(file_group, output_dir, target_scenario=None):
    """
    Merges a group of JSON files sharing the same scenario.
    
    Args:
        file_group: List of file paths to merge
        output_dir: Directory to save the merged file
        target_scenario: If provided, only print debug information for this scenario
    """
    if not file_group:
        return

    # Extract scenario from first file for output filtering
    first_file = os.path.basename(file_group[0])
    _, scenario_from_path, _ = extract_parts_from_filename(first_file)
    should_log = target_scenario is None or target_scenario.lower() in scenario_from_path.lower()

    # DEBUG: Print all files in group at the start
    if should_log:
        print("\nDEBUG - merge_json_files - Files to process:")
        for i, filepath in enumerate(file_group):
            print(f"  {i+1}. {filepath}")
        print("")

    merged_data = {
        'metadata': {
            # Store metadata per source file, keyed by the file's suffix
            'individual_metadata': {},
            'source_files': []
        },
        'dialogue': {}
    }
    output_filename_base = ""

    # Determine output filename base and collect metadata
    first_file_processed = False
    for filepath in file_group:
        filename = os.path.basename(filepath)
        acronym, scenario, suffix = extract_parts_from_filename(filename)
        
        # Only show debug output for target scenario
        should_log_file = should_log
        
        if should_log_file:
            print(f"DEBUG - Processing metadata for: {filename}")
            print(f"  DEBUG - Parsed parts: acronym='{acronym}', scenario='{scenario}', suffix='{suffix}'")

        # UNCOMMENT Check
        if acronym is None:
            if should_log_file:
                print(f"  DEBUG - ⚠️ SKIPPING file: {filename} - Could not parse filename")
            continue # Skip files that couldn't be parsed

        if not first_file_processed:
             # Use the first valid file's acronym and scenario for the merged filename
             output_filename_base = f"{acronym}_{scenario}"
             first_file_processed = True
             if should_log_file:
                 print(f"  DEBUG - Set output_filename_base to: {output_filename_base}")

        # Track source files
        merged_data['metadata']['source_files'].append(filename)
        if should_log_file:
            print(f"  DEBUG - Added to source_files list. Current count: {len(merged_data['metadata']['source_files'])}")

        # Determine the key for this file's metadata (usually the suffix)
        metadata_key = suffix

        # Load and collect metadata
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if should_log_file:
                    print(f"  DEBUG - Successfully loaded JSON from: {filename}")

            meta = data.get('metadata', {})
            individual_meta = {}
            # Use .get with default to avoid error if key missing
            individual_meta['synopsis'] = meta.get('synopsis', '')
            individual_meta['how_to_trigger'] = meta.get('how_to_trigger', '')
            if should_log_file:
                print(f"  DEBUG - Extracted metadata: synopsis={bool(individual_meta['synopsis'])}, how_to_trigger={bool(individual_meta['how_to_trigger'])}")

            # Only add if there's *any* content, even empty strings from .get
            if metadata_key in merged_data['metadata']['individual_metadata']:
                 if should_log_file:
                     print(f"  DEBUG - ⚠️ Warning: Metadata key '{metadata_key}' already exists for group '{output_filename_base}'. Overwriting.")
            merged_data['metadata']['individual_metadata'][metadata_key] = individual_meta
            if should_log_file:
                print(f"  DEBUG - Added metadata with key: '{metadata_key}'")

        except json.JSONDecodeError:
             if should_log_file:
                 print(f"  DEBUG - ⚠️ ERROR: Failed to decode JSON from {filepath}. SKIPPING file in metadata phase.")
             continue
        except Exception as e:
             if should_log_file:
                 print(f"  DEBUG - ⚠️ ERROR: Failed to read metadata from {filepath}: {e}. SKIPPING file in metadata phase.")
             continue

    if not output_filename_base:
         if should_log:
             print(f"DEBUG - ⚠️ CRITICAL: Could not determine base filename for group in {output_dir}. Skipping entire group.")
         return

    if should_log:
        print(f"\nDEBUG - Metadata phase complete. Proceeding to dialogue processing phase.")
        print(f"DEBUG - Source files recorded in metadata: {len(merged_data['metadata']['source_files'])}")
        print(f"DEBUG - Files marked for processing: {merged_data['metadata']['source_files']}")

    # Process and merge dialogue nodes
    files_processed = 0
    files_skipped = 0
    empty_dialogue_files = []  # Track files skipped due to empty dialogue
    for filepath in file_group:
        filename = os.path.basename(filepath)
        should_log_file = should_log
        
        if should_log_file:
            print(f"\nDEBUG - Processing dialogue for: {filename}")
            
        acronym, scenario, suffix = extract_parts_from_filename(filename)

        if acronym is None:
            if should_log_file:
                print(f"  DEBUG - ⚠️ SKIPPING file in dialogue phase: {filename} - Could not parse filename")
            files_skipped += 1
            continue

        id_prefix = suffix # Use the unique suffix as the prefix for node IDs
        if should_log_file:
            print(f"  DEBUG - Using ID prefix: '{id_prefix}'")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if should_log_file:
                    print(f"  DEBUG - Successfully loaded JSON from: {filename}")

            dialogue_data = data.get('dialogue', {})
            if should_log_file:
                print(f"  DEBUG - Dialogue data has {len(dialogue_data)} nodes before processing.") # Clarified log
            
            # THIS IS THE KEY CHECK - is dialogue empty?
            if not dialogue_data:
                if should_log_file:
                    print(f"  DEBUG - ⚠️ File has empty dialogue data: {filename} - Including it anyway")
                empty_dialogue_files.append(filename)
                # NOTE: We're now continuing processing despite empty dialogue
                # Instead of skipping with continue, we process the file anyway
                # This ensures the file appears in source_files
                
                # Still count the file as processed (not skipped) since we're not skipping it anymore
                files_processed += 1
                # Skip the node processing part since there are no nodes
                continue

            # Get ALL IDs of all nodes in the current file *before* modifying them
            # original_node_ids = set(dialogue_data.keys()) # Old method
            all_original_node_ids_in_file = set()
            if isinstance(dialogue_data, dict):
                for top_level_node_content in dialogue_data.values():
                    _collect_all_node_ids_recursively(top_level_node_content, all_original_node_ids_in_file)
            
            if should_log_file:
                # print(f"  DEBUG - Original node IDs (top-level keys): {original_node_ids}") # Old log line
                print(f"  DEBUG - All original node IDs collected from structure: {all_original_node_ids_in_file}")

            # Make a deep copy to avoid modifying the original data dict while iterating
            dialogue_data_copy = copy.deepcopy(dialogue_data)
            nodes_added = 0

            for node_id, node_data in dialogue_data_copy.items():
                # Update IDs recursively within this node structure
                update_node_ids_recursive(node_data, id_prefix, all_original_node_ids_in_file) # Use the comprehensive set

                # Add the processed node (with updated IDs) to the merged dialogue
                new_node_key = node_data['id'] # The ID should now be prefixed
                if new_node_key in merged_data['dialogue']:
                    # This warning handles potential node ID collisions
                    if should_log_file:
                        print(f"  DEBUG - ⚠️ Warning: Duplicate node ID '{new_node_key}' encountered while merging '{filename}'. Overwriting.")

                merged_data['dialogue'][new_node_key] = node_data # Use the new prefixed ID as the key
                nodes_added += 1

            if should_log_file:
                print(f"  DEBUG - ✅ Successfully processed {nodes_added} nodes from {filename}")
            files_processed += 1

        except json.JSONDecodeError:
             if should_log_file:
                 print(f"  DEBUG - ⚠️ ERROR: Failed to decode JSON from {filepath}. SKIPPING file in dialogue phase.")
             files_skipped += 1
             continue
        except Exception as e:
            if should_log_file:
                print(f"  DEBUG - ⚠️ ERROR: Failed to process dialogue from {filepath}: {e}. SKIPPING file in dialogue phase.")
            files_skipped += 1
            continue

    if should_log:
        print(f"\nDEBUG - Dialogue processing complete:")
        print(f"  - Total files processed successfully: {files_processed}")
        print(f"  - Total files skipped: {files_skipped}")
        if empty_dialogue_files:
            print(f"  - Files skipped due to empty dialogue: {len(empty_dialogue_files)}")
            for f in empty_dialogue_files:
                print(f"    - {f}")
        print(f"  - Total source files recorded in metadata: {len(merged_data['metadata']['source_files'])}")
        print(f"  - Total dialogue nodes in merged result: {len(merged_data['dialogue'])}")

    # Construct output path
    output_path = os.path.join(output_dir, f"{output_filename_base}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save merged file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2)
        if should_log:
            print(f"DEBUG - ✅ Successfully wrote merged file: {output_path}")
    except Exception as e:
        if should_log:
            print(f"DEBUG - ⚠️ ERROR writing merged file {output_path}: {e}")

    return output_path, merged_data


def process_directory(input_root, output_root, target_scenario="LaezelRecruitment"):
    """
    Walks through the input directory, groups files by scenario, and merges them.
    
    Args:
        input_root: Root directory of original JSON files
        output_root: Root directory for merged output
        target_scenario: Only show debug output for this scenario
    """
    # Group files by directory and LOWERCASE scenario name to handle case variations
    # Key: (directory_path, lowercase_scenario_name), Value: list of filepaths
    scenario_groups = defaultdict(list)
    # Keep track of the original case for each scenario to use in output filenames
    scenario_original_case = {}

    for dirpath, _, filenames in os.walk(input_root):
        # Determine the corresponding output directory path
        relative_path = os.path.relpath(dirpath, input_root)
        current_output_dir = os.path.join(output_root, relative_path)

        for filename in filenames:
            if filename.endswith('.json'):
                filepath = os.path.join(dirpath, filename)
                acronym, scenario, suffix = extract_parts_from_filename(filename)

                if scenario: # Only group if scenario could be extracted
                     # Convert scenario to lowercase for grouping to handle case variations
                     lowercase_scenario = scenario.lower()
                     
                     # Track original case for output filenames (first occurrence)
                     if lowercase_scenario not in scenario_original_case:
                         scenario_original_case[lowercase_scenario] = scenario
                     
                     # Group by the directory path and the LOWERCASE scenario identifier
                     group_key = (current_output_dir, lowercase_scenario)
                     # Only print debug for target scenario (case-insensitive)
                     if target_scenario is None or target_scenario.lower() in lowercase_scenario:
                         print(f"  Debug: Adding '{filename}' to group {group_key}") # DEBUG PRINT
                     scenario_groups[group_key].append(filepath)

    # Merge each group
    merged_files = []
    total_files_processed = 0
    
    for (output_dir_path, lowercase_scenario), filepaths in scenario_groups.items():
         # Use the original case for output filenames
         original_case_scenario = scenario_original_case[lowercase_scenario]
         
         # DEBUG PRINT only for target scenario (case-insensitive)
         show_debug = target_scenario is None or target_scenario.lower() in lowercase_scenario
         if show_debug:
             print(f"\nDebug: Files in group ('{output_dir_path}', '{original_case_scenario}' [case-insensitive]):")
             for fp in filepaths:
                 print(f"    - {os.path.basename(fp)}")
         # --- END DEBUG ---
         if len(filepaths) > 0: # Only merge if there are files in the group
             if show_debug:
                 print(f"Merging scenario '{original_case_scenario}' in '{os.path.relpath(output_dir_path, output_root)}' ({len(filepaths)} files)")
             else:
                 print(f"Merging scenario '{original_case_scenario}' in '{os.path.relpath(output_dir_path, output_root)}' ({len(filepaths)} files)")
             
             # Preserve the original scenario case for the first file to maintain output filename consistency
             first_file = os.path.basename(filepaths[0])
             first_acronym, _, _ = extract_parts_from_filename(first_file)
             
             # Force the output filename to use the original case from the first file encountered
             result = merge_json_files(filepaths, output_dir_path, target_scenario)
             if result:
                 output_path, merged_data = result
                 merged_files.append(output_path)
                 total_files_processed += len(merged_data['metadata']['source_files'])
    
    if target_scenario:
        print(f"\nDEBUG - Summary (showing only for {target_scenario}):")
    else:
        print(f"\nDEBUG - Summary:")
    print(f"  - Total merged files created: {len(merged_files)}")
    print(f"  - Total original files processed: {total_files_processed}")
    
    return merged_files


def main():
    input_folder = 'output' # Folder with parsed JSON files
    output_folder = 'output_merged' # Folder for merged JSON files

    print(f"Starting merge process...")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")

    if not os.path.isdir(input_folder):
        print(f"ERROR: Input directory '{input_folder}' not found.")
        return False

    if os.path.exists(output_folder):
        print(f"INFO: Output directory '{output_folder}' already exists. Existing files might be overwritten.")
    else:
        os.makedirs(output_folder)

    merged_files = process_directory(input_folder, output_folder)
    
    if not merged_files:
        print("ERROR: No merged files were created! Something went wrong.")
        return False
        
    print(f"\nMerge process finished successfully.")
    print(f"Created {len(merged_files)} merged files.")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        import sys
        sys.exit(1) 