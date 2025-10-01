import re
import sys
import json

def parse_osiris_dialogs(script_content: str) -> list[str]:
    """
    Parses an Osiris script content and extracts unique dialogue names 
    in chronological order of first appearance.
    Looks for (DIALOGRESOURCE) markers and potential dialog names passed as arguments.
    """
    ordered_dialogs = []
    
    # Pattern for likely dialog resource identifiers (alphanumeric, underscore, hyphen, optionally ending in GUID)
    dialog_id_pattern = r'[a-zA-Z0-9_-]+(?:_[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12})?'
    
    # Pattern 1: Explicit (DIALOGRESOURCE)MyDialogName
    explicit_pattern = re.compile(r'(\((?:DIALOGRESOURCE)\)\s*(' + dialog_id_pattern + r')\b)') # Capture group 2 is the name
    
    # Pattern 2: Potential dialog names in argument-like positions (no lookbehind)
    # Capture group 1 is the name
    argument_pattern = re.compile(r'[,\(]\s*(' + dialog_id_pattern + r')\b(?=\s*[,\)])') 

    lines = script_content.splitlines()
    for line_number, line in enumerate(lines, 1):
        # Keep track of spans covered by explicit (DIALOGRESOURCE) matches on this line
        explicit_spans = []

        # First pass: Find explicit dialogs
        for match in explicit_pattern.finditer(line):
            full_match_text = match.group(1) # The full matched text like "(DIALOGRESOURCE)Name"
            dialog_name = match.group(2)     # Just the name
            span = match.span(1)             # Position of the full match
            
            if dialog_name and dialog_name != "NULL_00000000-0000-0000-0000-000000000000":
                if dialog_name not in ordered_dialogs:
                    to_add = dialog_name.rsplit("_", 1)[0]
                    if to_add not in ordered_dialogs:
                        ordered_dialogs.append(to_add)
                explicit_spans.append(span) # Store the span of the full match

        # Second pass: Find potential argument dialogs
        for match in argument_pattern.finditer(line):
            dialog_name = match.group(1) # The potential dialog name
            match_span = match.span(1)   # The span of the dialog name itself
            
            # Check if this match overlaps with any explicit spans found earlier
            is_overlapped = False
            for start, end in explicit_spans:
                # Check for any overlap between match_span and the explicit span
                if max(start, match_span[0]) < min(end, match_span[1]):
                    is_overlapped = True
                    break
            
            if not is_overlapped:
                # Ensure a name was captured and it's not a common null placeholder or a simple variable
                if dialog_name and not dialog_name.startswith('_') and dialog_name != "NULL_00000000-0000-0000-0000-000000000000":
                    # Add to list only if it's the first time seeing this dialog
                    if dialog_name not in ordered_dialogs:
                        to_add = dialog_name.rsplit("_", 1)[0]
                        if to_add not in ordered_dialogs:
                            ordered_dialogs.append(to_add)
                        
    # Filter out potential false positives (like flags caught by pattern 2)
    # This is heuristic: dialogs often contain GUIDs or specific keywords
    final_dialogs = []
    known_dialog_keywords = ["_AD_", "_Scene_", "_Recruitment_", "Dialog", "GUID"]
    guid_pattern = re.compile(r'_[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}')
    for name in ordered_dialogs:
        is_likely_dialog = False
        if guid_pattern.search(name):
            is_likely_dialog = True
        else:
            for keyword in known_dialog_keywords:
                if keyword in name:
                    is_likely_dialog = True
                    break
        # Add more specific checks if needed, e.g., checking against known function names
        
        if is_likely_dialog:
            final_dialogs.append(name)
            
    return final_dialogs

def parse_osiris_flags(script_content: str) -> list[str]:
    """
    Parses an Osiris script content and extracts unique flag names in chronological order of first appearance.

    Flags are extracted from lines containing the word "flag" (case-insensitive).
    It identifies flags explicitly marked like (FLAG)MyFlagName or passed as arguments
    to known flag-handling functions.
    """
    ordered_flags = []
    
    # Pattern for valid flag identifiers (alphanumeric, underscore, hyphen)
    flag_id_pattern = r'[a-zA-Z0-9_-]+'
    
    # Compiled regular expressions for efficiency
    # Each pattern is designed to capture the flag name in its first group.
    regex_patterns = [
        # Pattern 1: Explicit (FLAG)MyFlag, (Flag)MyFlag, (flag)MyFlag in various contexts.
        # This is the most common and direct way flags are specified.
        re.compile(r'\((?:FLAG|Flag|flag)\)(' + flag_id_pattern + r')\b'),
        
        # --- Temporarily disabling other patterns for debugging ---
        # Pattern 2: Functions known to take a raw flag name as their first significant argument.
        # The negative lookahead (?!\((?:FLAG|Flag|flag)\)) ensures these patterns only match
        # "bare" flag names, not when they are already prefixed with (FLAG),
        # as Pattern 1 would have already captured the name from (FLAG)FlagName.
        # Added \s*\) to match optional space and the function's closing parenthesis.
        re.compile(r'ClearFlag\s*\(\s*(?!\((?:FLAG|Flag|flag)\))(' + flag_id_pattern + r')\b\s*\)'),
        re.compile(r'GetFlag\s*\(\s*(?!\((?:FLAG|Flag|flag)\))(' + flag_id_pattern + r')\b\s*\)'),
        re.compile(r'PROC_GlobalSetFlagAndCache\s*\(\s*(?!\((?:FLAG|Flag|flag)\))(' + flag_id_pattern + r')\b\s*\)'),
        re.compile(r'PROC_GlobalClearFlagAndCache\s*\(\s*(?!\((?:FLAG|Flag|flag)\))(' + flag_id_pattern + r')\b\s*\)'),
        re.compile(r'FlagSet\s*\(\s*(?!\((?:FLAG|Flag|flag)\))(' + flag_id_pattern + r')\b\s*\)'),
        re.compile(r'SetFlag\s*\(\s*(?!\((?:FLAG|Flag|flag)\))(' + flag_id_pattern + r')\b\s*\)'),
        re.compile(r'DB_GlobalFlag\s*\(\s*(?!\((?:FLAG|Flag|flag)\))(' + flag_id_pattern + r')\b\s*\)'),
    ]

    lines = script_content.splitlines()
    for line_number, line in enumerate(lines, 1):
        # User's condition: Only process lines containing the word "flag".
        # Using word boundaries (\b) to ensure "flag" is a whole word.
        #-- Removing this check as it might be too strict and filter out valid flag lines --#
        # if not re.search(r'\\\\bflag\\\\b', line, re.IGNORECASE):
        #     continue

        for pattern in regex_patterns:
            matches_on_line = pattern.findall(line)
            for flag_name in matches_on_line:
                # Ensure a flag name was actually captured and it's not a common null placeholder.
                if flag_name and flag_name != "NULL_00000000-0000-0000-0000-000000000000":
                    processed_flag_name = flag_name.rsplit("_", 1)[0]
                    # Add to list only if it's the first time seeing this flag
                    if processed_flag_name not in ordered_flags:
                        ordered_flags.append(processed_flag_name)
    
    return ordered_flags

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Default to the filename mentioned in the problem description if no argument is provided.
        filepath = "Goals/Act1_CHA_LaezelRecruit.txt"#"Goals/Act1_CRA_AstarionRecruitment.txt"#"Goals/Act1_ORI_Gale_PostEA.txt"#"Goals/Act1_UND_SharFort.txt"#"Goals/Act1_DEN_Training.txt"#"Goals/Act1_PLA_ZhentShipment.txt"#"Goals/Act1_CHA_LaezelRecruit.txt"#"Goals/Act1_CRA_AstarionRecruitment.txt"#
        prefix = filepath.rsplit("/", 1)[1].replace(".txt", "").split("_")[1] + "_"
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            script_content = f.read()
        
        osiris_flags = parse_osiris_flags(script_content)
        osiris_dialogs = parse_osiris_dialogs(script_content)
        filtered_osiris_dialogs = []
        with open("all_sessions.json", "r", encoding="utf-8") as f:
            all_sessions = json.load(f)
        for osiris_dialog in osiris_dialogs:
            for session in all_sessions:
                session_name = session.rsplit("/", 1)[1].replace(".json", "")
                if osiris_dialog == session_name:
                    filtered_osiris_dialogs.append(session)
        osiris_dialogs = filtered_osiris_dialogs
        # Print Dialogs
        #if osiris_dialogs:
        #    print(f"\nFound {len(osiris_dialogs)} unique dialogs in chronological order in '{filepath}':")
        #    print(f"{osiris_dialogs}")
        #else:
        #    print(f"\nNo dialogs found in '{filepath}' matching the specified criteria.")

        # Print Flags
        if osiris_flags:
            #print(f"Found {len(osiris_flags)} unique flags in chronological order in '{filepath}':")
            # Print flags on one line for brevity
            #print(f"{osiris_flags}")

            # --- Find matching JSON files ---
            all_flags_path = "all_flags.json"
            session_data = [] # List to hold data for sorting

            try:
                with open(all_flags_path, "r", encoding="utf-8") as f_json:
                    all_json_flags_data = json.load(f_json)
                
                #print(f"\nSearching for flags from '{filepath}' in '{all_flags_path}'...")

                # Create index lookups for faster searching
                dialog_name_to_index = {name.rsplit("/", 1)[-1].replace(".json", ""): i for i, name in enumerate(osiris_dialogs)}
                flag_name_to_index = {name: i for i, name in enumerate(osiris_flags)}

                # Identify all unique session paths involved
                all_involved_paths = set(osiris_dialogs) | set(all_json_flags_data.keys())

                for session_path in all_involved_paths:
                    session_name = session_path.rsplit("/", 1)[-1].replace(".json", "")
                    flags_in_session = all_json_flags_data.get(session_path, [])

                    # Find first associated dialog index
                    # Use dialog_name_to_index, checking if the session_name (base name) exists as a key
                    first_dialog_idx = dialog_name_to_index.get(session_name, float('inf'))
                    
                    # Find first associated prefix flag index
                    first_prefix_flag_idx = float('inf')
                    for flag in flags_in_session:
                        if flag.startswith(prefix):
                            idx = flag_name_to_index.get(flag, float('inf'))
                            first_prefix_flag_idx = min(first_prefix_flag_idx, idx)

                    # Find first associated other flag index
                    first_other_flag_idx = float('inf')
                    for flag in flags_in_session:
                        if not flag.startswith(prefix):
                            idx = flag_name_to_index.get(flag, float('inf'))
                            first_other_flag_idx = min(first_other_flag_idx, idx)

                    # Only add if it's associated with *something* from the Osiris script
                    if first_dialog_idx != float('inf') or first_prefix_flag_idx != float('inf') or first_other_flag_idx != float('inf'):
                        session_data.append({
                            "path": session_path,
                            "dialog_idx": first_dialog_idx,
                            "prefix_flag_idx": first_prefix_flag_idx,
                            "other_flag_idx": first_other_flag_idx
                        })
                
                # --- Sort the sessions --- 
                # Sort by dialog index, then prefix flag index, then other flag index, then path for stability
                session_data.sort(key=lambda x: (
                    x["prefix_flag_idx"], # Primary sort key
                    x["dialog_idx"],        # Secondary sort key
                    x["other_flag_idx"],    # Tertiary sort key
                    x["path"] # Final tie-breaker for stability
                ))

                # --- Print the results ---
                if session_data:
                    print(f"\n--- Ordered JSON files (Sorted by Prefix ('{prefix}') Flag > Dialog > Other Flag) ---")

                    hierarchical_indices = []
                    if not session_data: # Should be caught by outer if, but defensive
                        pass
                    else:
                        # Initialize for the first item
                        h_main, h_sub, h_subsub = 1, 1, 1
                        hierarchical_indices.append((h_main, h_sub, h_subsub))

                        # Calculate for subsequent items
                        for k_idx in range(1, len(session_data)):
                            prev_data_item = session_data[k_idx-1]
                            curr_data_item = session_data[k_idx]
                            
                            # Get the hierarchical index of the *previous* item to base increments on
                            prev_h_main, prev_h_sub, prev_h_subsub = hierarchical_indices[k_idx-1]

                            # Priority 1: Prefix Flag Index
                            if curr_data_item['prefix_flag_idx'] != prev_data_item['prefix_flag_idx']:
                                h_main = prev_h_main + 1
                                h_sub = 1
                                h_subsub = 1
                            # Priority 2: Dialog Index
                            elif curr_data_item['dialog_idx'] != prev_data_item['dialog_idx']:
                                h_main = prev_h_main # Stays same
                                h_sub = prev_h_sub + 1
                                h_subsub = 1
                            # Priority 3: Other Flag Index
                            elif curr_data_item['other_flag_idx'] != prev_data_item['other_flag_idx']:
                                h_main = prev_h_main # Stays same
                                h_sub = prev_h_sub   # Stays same
                                h_subsub = prev_h_subsub + 1 # Increment for distinct item
                            
                            hierarchical_indices.append((h_main, h_sub, h_subsub))

                        # Now print using these hierarchical_indices
                        for k_idx, data_item_to_print in enumerate(session_data):
                            h_main_val, h_sub_val, h_subsub_val = hierarchical_indices[k_idx]
                            enumeration_str = f"{h_main_val}-{h_sub_val}-{h_subsub_val}"
                            
                            detail = f"dialog_idx={data_item_to_print['dialog_idx']}, prefix_flag_idx={data_item_to_print['prefix_flag_idx']}, other_flag_idx={data_item_to_print['other_flag_idx']}"
                            detail = detail.replace("float('inf')", "N/A")
                            print(f"{enumeration_str}. {data_item_to_print['path']} ({detail})")
                else:
                    print("\nNo relevant session files found based on the Osiris script.")
                    
            except FileNotFoundError:
                print(f"\nError: Could not find '{all_flags_path}'. Please run parse_every_flag.py first.")
            except json.JSONDecodeError:
                print(f"\nError: Could not decode JSON from '{all_flags_path}'.")
            except Exception as e_json:
                print(f"\nAn error occurred while processing '{all_flags_path}': {e_json}")
            # --- End find matching JSON files ---

        else:
            print(f"No flags found in '{filepath}' matching the specified criteria.")
            
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        print("Please provide a valid path as a command-line argument or ensure the default file exists.")
    except Exception as e:
        print(f"An error occurred: {e}") 