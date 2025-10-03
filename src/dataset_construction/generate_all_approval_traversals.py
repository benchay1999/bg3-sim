import json
import os
import sys
import itertools
from colorama import init, Fore, Style
# Ensure the import path is correct if running as a script vs module
try:
    from .scenario_simulator import ScenarioSimulator
except ImportError:
    from scenario_simulator import ScenarioSimulator


init(autoreset=True)

# Helper to count utterances (can be shared)
def count_utterances(session_sim, path):
    if not path: return 0
    utterance_count = 0
    for node_id in path:
        if node_id in ["MAX_DEPTH_REACHED", "NODE_NOT_FOUND"]: continue
        node = session_sim._get_node(node_id)
        if node and node.get('text'): utterance_count += 1
    return utterance_count

def find_approval_paths_for_session(simulator, session_id, min_utterances=3):
    """
    Find all paths for a given session that contain at least one approval node
    and meet the minimum utterance requirement.
    """
    session_sim = simulator.session_simulators.get(session_id)
    if not session_sim:
        print(f"{Fore.RED}Simulator not found for session {session_id} during approval path search.{Style.RESET_ALL}")
        return []

    paths = simulator.session_path_options.get(session_id, [])
    if not paths:
        return []

    # Find nodes with approval effects within this session
    approval_nodes = []
    for node_id, node_data in session_sim.all_nodes.items():
         if node_data.get('approval') and len(node_data.get('approval')) > 0:
             approval_nodes.append(node_id)

    if not approval_nodes:
        return [] # No approval nodes in this session

    valid_approval_paths = []
    for path in paths:
        visits_approval_node = any(node_id in path for node_id in approval_nodes)
        if visits_approval_node:
            utterances = count_utterances(session_sim, path)
            if utterances >= min_utterances:
                valid_approval_paths.append(path)

    return valid_approval_paths

def find_all_paths_meeting_min_utterances(simulator, session_id, min_utterances=3):
    """
    Find all paths for a given session that meet the minimum utterance requirement,
    regardless of approval status.
    """
    session_sim = simulator.session_simulators.get(session_id)
    if not session_sim:
        print(f"{Fore.RED}Simulator not found for session {session_id} during path search.{Style.RESET_ALL}")
        return []

    paths = simulator.session_path_options.get(session_id, [])
    if not paths:
        return []

    valid_paths = []
    for path in paths:
        utterances = count_utterances(session_sim, path)
        if utterances >= min_utterances:
            valid_paths.append(path)

    return valid_paths

def generate_all_approval_traversals(scenario_file, output_json_file, min_utterances=3):
    """
    Generates and exports all possible scenario traversals based on constraints.
    For each session in a traversal, it prioritizes paths with approval nodes.
    If a session has no approval paths meeting min_utterances, it uses any path
    meeting min_utterances.
    """
    try:
        simulator = ScenarioSimulator(scenario_file)
    except FileNotFoundError:
        print(f"{Fore.RED}Error: Scenario file not found at {scenario_file}{Style.RESET_ALL}")
        return
    except json.JSONDecodeError:
        print(f"{Fore.RED}Error: Invalid JSON in scenario file {scenario_file}{Style.RESET_ALL}")
        return
    except Exception as e:
        print(f"{Fore.RED}Error initializing ScenarioSimulator: {e}{Style.RESET_ALL}")
        return

    print(f"{Fore.CYAN}Simulating all sessions to find paths...{Style.RESET_ALL}")
    for session_id in simulator.session_ids:
        try:
            simulator._simulate_session(session_id) # Populates session_path_options
        except Exception as e:
            print(f"{Fore.RED}Error simulating session {session_id}: {e}{Style.RESET_ALL}")
            # Consider if simulation errors should halt the process
            # return # Optional: Abort on simulation error

    print(f"{Fore.CYAN}Generating valid session sequences...{Style.RESET_ALL}")
    valid_sequences = simulator._generate_valid_sequences(
        prioritize_approval=False, # We handle priority during path selection
        include_all_sessions=True
    )

    if not valid_sequences:
        print(f"{Fore.YELLOW}No valid session sequences found based on constraints.{Style.RESET_ALL}")
        return

    print(f"Found {len(valid_sequences)} valid session sequences. Analyzing paths for traversals...")

    all_final_traversals = []
    processed_sequences = 0
    skipped_sequences_count = 0

    for sequence in valid_sequences:
        sequence_path_options = {}
        possible_to_form_traversal = True
        for session_id in sequence:
            # 1. Try to find approval paths first
            paths_for_session = find_approval_paths_for_session(simulator, session_id, min_utterances)

            if not paths_for_session:
                # 2. If no approval paths, find *any* paths meeting min utterances
                # print(f"{Fore.YELLOW}  Session {session_id}: No approval paths found/suitable. Checking all paths...{Style.RESET_ALL}")
                paths_for_session = find_all_paths_meeting_min_utterances(simulator, session_id, min_utterances)

            if not paths_for_session:
                # 3. If still no paths found (neither approval nor general meeting criteria),
                # this sequence cannot be completed.
                print(f"{Fore.YELLOW}  Sequence {sequence}: Cannot find any suitable paths (min {min_utterances} utterances) for session '{session_id}'. Skipping sequence.{Style.RESET_ALL}")
                possible_to_form_traversal = False
                break # Stop processing this sequence

            # Store the selected list of paths (either approval or all valid ones)
            sequence_path_options[session_id] = paths_for_session
            # print(f"  Session {session_id}: Using {len(paths_for_session)} paths for combinations.")

        if not possible_to_form_traversal:
            skipped_sequences_count += 1
            continue # Move to the next sequence

        # If we reached here, all sessions in the sequence have at least one suitable path.
        processed_sequences += 1

        # Prepare lists of paths for itertools.product
        path_lists = [sequence_path_options[session_id] for session_id in sequence]

        # Generate all combinations of paths for this sequence
        path_combinations_iter = itertools.product(*path_lists)

        num_combinations_for_sequence = 0
        for path_combo in path_combinations_iter:
            num_combinations_for_sequence += 1
            traversal = {
                "session_sequence": sequence,
                "paths": dict(zip(sequence, path_combo)),
            }
            all_final_traversals.append(traversal)

        # Avoid overly verbose logging if many combinations exist
        # print(f"  Sequence {sequence}: Generated {num_combinations_for_sequence} traversals.")
        if num_combinations_for_sequence > 0:
             print(f"Processed sequence {sequence} -> {num_combinations_for_sequence} traversals generated.")


    print(f"\n{Fore.GREEN}Processing complete.{Style.RESET_ALL}")
    print(f" - Analyzed {len(valid_sequences)} valid sequences based on constraints.")
    print(f" - Found {processed_sequences} sequences where all sessions had suitable paths (prioritizing approval)." if processed_sequences > 0 else "- No sequences could be fully processed.")
    if skipped_sequences_count > 0:
        print(f" - Skipped {skipped_sequences_count} sequences because at least one session lacked any suitable paths (min {min_utterances} utterances)." )
    print(f" - Generated a total of {Fore.CYAN}{len(all_final_traversals)}{Style.RESET_ALL} traversals.")

    # Export to JSON
    if all_final_traversals:
        print(f"\nExporting {len(all_final_traversals)} traversals to {output_json_file}...")
        output_data = {
            "scenario_source_file": os.path.basename(scenario_file),
            "scenario_name": simulator.scenario_name,
            "min_utterances_per_path": min_utterances,
            "total_traversals_generated": len(all_final_traversals),
            "traversals": all_final_traversals
        }
        try:
            with open(output_json_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"{Fore.GREEN}Export successful to {output_json_file}{Style.RESET_ALL}")
        except IOError as e:
            print(f"{Fore.RED}Error writing output file {output_json_file}: {e}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}An unexpected error occurred during export: {e}{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}No traversals generated, skipping export.{Style.RESET_ALL}")


if __name__ == "__main__":
    print(f"{Fore.CYAN}BG3 All Traversals Generator (Prioritizing Approval){Style.RESET_ALL}")

    # --- Argument parsing --- (rest of the main block remains the same)
    if len(sys.argv) > 1:
        scenario_file = sys.argv[1]
    else:
        # Adjust default path if necessary
        default_scenario = "../../output_merged/Act1/Chapel/cha_outside.json"
        try:
            # Check relative path from script location first
            script_dir = os.path.dirname(__file__)
            potential_default = os.path.join(script_dir, default_scenario)
            if os.path.exists(potential_default):
                default_scenario = potential_default
            else:
                 # Fallback to path relative to CWD if not found near script
                 potential_default = "output_merged/Act1/Chapel/cha_outside.json"
                 if os.path.exists(potential_default):
                      default_scenario = potential_default
                 else:
                     # If neither exists, use the original hardcoded string as placeholder
                     default_scenario = "output_merged/Act1/Chapel/cha_outside.json"
        except NameError: # __file__ is not defined (e.g. interactive) 
             default_scenario = "output_merged/Act1/Chapel/cha_outside.json"

        scenario_file_input = input(f"Enter path to scenario file [default: {default_scenario}]: ")
        scenario_file = scenario_file_input or default_scenario

    if not os.path.isfile(scenario_file):
        print(f"{Fore.RED}Error: Input file not found: {scenario_file}{Style.RESET_ALL}")
        sys.exit(1)

    # Default output file name based on input
    scenario_dir = os.path.dirname(scenario_file) or '.' # Handle case where file is in CWD
    scenario_base_name = os.path.splitext(os.path.basename(scenario_file))[0]
    default_output_file = os.path.join(scenario_dir, f"{scenario_base_name}_all_priority_approval_traversals.json")

    output_file = input(f"Enter output JSON file name [default: {default_output_file}]: ") or default_output_file

    min_utterances = 3
    try:
         min_utterances_input = input(f"Minimum utterances per session path [default: 3]: ")
         if min_utterances_input:
             min_utterances = int(min_utterances_input)
             if min_utterances < 0: min_utterances = 0
    except ValueError:
         print(f"{Fore.YELLOW}Invalid input for minimum utterances. Using default value of 3.{Style.RESET_ALL}")

    generate_all_approval_traversals(scenario_file, output_file, min_utterances) 