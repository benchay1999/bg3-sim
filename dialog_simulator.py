import json
import sys
import os
import numpy as np
from colorama import init, Fore, Back, Style
import random
import traceback # For error handling during rendering
import time

# Default dialog JSON path used by the simulator
DEFAULT_DIALOG_JSON = 'output/Act2/MoonriseTowers/MOO_Jailbreak_Wulbren.json'

# Load OpenAI API key from .env file
from dotenv import load_dotenv
load_dotenv()

# Optional LiteLLM import for LLM-based context generation
try:
    from litellm import completion as litellm_completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

# Initialize colorama for colored terminal output
init()

# Add a check for graphviz import success
try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print(f"{Fore.YELLOW}Warning: 'graphviz' Python package not found. Visualization features will be disabled.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Install it with: pip install graphviz{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}You also need to install the Graphviz software: https://graphviz.org/download/{Style.RESET_ALL}")

# Optional FAISS + embedding model for RAG
try:
    import faiss  # faiss-cpu
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print(f"{Fore.YELLOW}Warning: 'faiss' (faiss-cpu) is not available. RAG features will be disabled.{Style.RESET_ALL}")

# Prefer OpenAI embeddings via LiteLLM if available; fallback to sentence-transformers
try:
    from litellm import embedding as litellm_embedding
    LITELLM_EMBED_AVAILABLE = True
except ImportError:
    LITELLM_EMBED_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENT_EMBED_AVAILABLE = True
    DEFAULT_EMBED_MODEL = 'all-MiniLM-L6-v2'
except ImportError:
    SENT_EMBED_AVAILABLE = False
    print(f"{Fore.YELLOW}Warning: 'sentence-transformers' is not installed. RAG features will be disabled.{Style.RESET_ALL}")

# Set default embed model to OpenAI large if LiteLLM embeddings are available
if LITELLM_EMBED_AVAILABLE:
    DEFAULT_EMBED_MODEL = 'text-embedding-3-large'

class DialogSimulator:
    def __init__(self, json_file=DEFAULT_DIALOG_JSON):
        """Initialize the dialog simulator with the specified JSON file"""
        with open(json_file, 'r', encoding='utf-8') as f:
            self.dialog_tree = json.load(f)
        
        # Remember which raw file this simulator is using
        self.json_path = json_file

        self.root_nodes = {}
        self.metadata = self.dialog_tree["metadata"]
        self.all_nodes = self.dialog_tree["dialogue"]  # All nodes including children
        
        # Extract root nodes from the dialog tree
        self.root_nodes = {node_id: node_data for node_id, node_data in self.dialog_tree["dialogue"].items() 
                           if not self._is_child_node(node_id)}
                
        print(f"Loaded dialog tree with {len(self.all_nodes)} nodes, {len(self.root_nodes)} root nodes")
        
        # Keep track of the most recent LLM user input payload used for context generation
        self.last_llm_input = None

        # Companion states to track approval changes
        self.companion_approvals = {
            "Gale": 0,
            "Astarion": 0,
            "Lae'zel": 0,
            "Shadowheart": 0,
            "Wyll": 0,
            "Karlach": 0,
            "Halsin": 0,
            "Minthara": 0,
            "Minsc": 0
        }
        
        # Track history of approval changes with node IDs
        self.companion_approval_history = {
            "Gale": [],
            "Astarion": [],
            "Lae'zel": [],
            "Shadowheart": [],
            "Wyll": [],
            "Karlach": [],
            "Halsin": [],
            "Minthara": [],
            "Minsc": []
        }
        
        # Track visited nodes in a session
        self.visited_nodes = []
        
        # Track flags that have been set during playthrough
        self.default_flags = ["ORI_INCLUSION_GALE", "ORI_INCLUSION_ASTARION", "ORI_INCLUSION_LAEZEL", "ORI_INCLUSION_SHADOWHEART", "ORI_INCLUSION_WYLL", "ORI_INCLUSION_KARLACH", "ORI_INCLUSION_HALSIN", "ORI_INCLUSION_MINTHARA", "ORI_INCLUSION_MINSC", "ORI_INCLUSION_RANDOM"]
        self.active_flags = set(self.default_flags)

        # Lazy-initialized embeddings and FAISS index
        self._embedder = None
        self._faiss_index = None
        self._faiss_dim = None
        self._faiss_id_to_meta = {}
        self._faiss_path_to_id = {}
        self._faiss_path_to_ids = {}

        # Track last RAG retrieval for meta export
        self.last_retrieved_sessions = []  # list[str]
        self.last_retrieved_synopses = []  # list[str]
    
    def _get_effective_text(self, node_data):
        """Return best display text for a node without mutating input."""
        text = node_data.get('text', '')
        if text:
            return text
        return node_data.get('context', '')
    
    def set_initial_flags(self, flags):
        """Set the initial active flags for the simulator."""
        # Ensure flags is a set
        if isinstance(flags, set):
            self.active_flags = flags.copy() # Work with a copy
        elif isinstance(flags, (list, tuple)):
            self.active_flags = set(flags)
        else:
            print(f"{Fore.YELLOW}Warning: Invalid type for initial flags. Expected set, list, or tuple. Using defaults.{Style.RESET_ALL}")
            self.active_flags = set(self.default_flags) # Fallback
        # print(f"{Fore.BLUE}Initial flags set: {len(self.active_flags)}{Style.RESET_ALL}") # Optional debug
    
    def _is_child_node(self, node_id):
        """Check if a node is a child node of any other node"""
        for other_id, other_data in self.all_nodes.items():
            if other_id != node_id:
                children = other_data.get('children', {})
                if node_id in children:
                    return True
        return False
    
    def _get_node(self, node_id):
        """Get a node by its ID, searching in the entire dialog tree structure"""
        # First check in top-level nodes
        if node_id in self.all_nodes:
            return self.all_nodes[node_id]
        
        # If not found at top level, search in children recursively
        for _, node_data in self.all_nodes.items():
            if 'children' in node_data:
                result = self._find_node_in_children(node_id, node_data['children'])
                if result:
                    return result
        
        return None

    def _find_node_in_children(self, node_id, children):
        """Recursively search for a node in the children dictionary"""
        if node_id in children:
            return children[node_id]
        
        for _, child_data in children.items():
            if 'children' in child_data and child_data['children']:
                result = self._find_node_in_children(node_id, child_data['children'])
                if result:
                    return result
        
        return None
    
    def _process_approvals(self, node_data):
        """Process approval changes from a node"""
        node_id = node_data.get('id', '')
        for approval in node_data.get('approval', []):
            parts = approval.split()
            if len(parts) >= 2:
                char_name = ' '.join(parts[:-1])
                value = parts[-1]
                try:
                    # Handle approval values like "1" or "-1"
                    if char_name in self.companion_approvals:
                        approval_value = int(value)
                        # Update cumulative approval
                        self.companion_approvals[char_name] += approval_value
                        # Record in approval history with node ID
                        self.companion_approval_history[char_name].append({
                            "node_id": node_id,
                            "value": approval_value,
                            "text": node_data.get('text', ''),
                            "speaker": node_data.get('speaker', ''),
                            "context": node_data.get('context', '')
                        })
                except (ValueError, KeyError):
                    pass
    
    def _process_setflags(self, node_data):
        """Process flags that are set by a node"""
        
        # if setflags has a flag with " = False", remove it from the active flags
        for flag in node_data.get('setflags', []):
            if "= False" in flag:
                try:
                    self.active_flags.remove(flag.split('= False')[0].strip())
                except KeyError:
                    pass
            else:
                self.active_flags.add(flag.strip())
    
    def _check_flags(self, node_data):
        """Check if required flags are met for a node"""
        # A simple implementation - in a real game this would be more complex
        # if checkflags is empty, return True
        if not node_data.get('checkflags', []):
            return True
        # if checkflags has a flag with " = False", check whether it is not set.
        for flag in node_data.get('checkflags', []):
            if "= False" in flag:
                if flag.split('= False')[0].strip() in self.active_flags:
                    return False
            else:
                if flag.strip() not in self.active_flags:
                    return False
        return True
    
    def display_metadata(self):
        """Display the metadata"""
        print(f"\n{Fore.WHITE}===== METADATA ====={Style.RESET_ALL}")
        print(f"Synopsis: {self.metadata.get('synopsis', '')}")
        print(f"How to trigger: {self.metadata.get('how_to_trigger', '')}")
    
    def display_node(self, node_id, node_data):
        """Display a dialog node with formatting"""
        speaker = node_data.get('speaker', 'Unknown')
        text = node_data.get('text', '') or node_data.get('context', '')
        node_type = node_data.get('node_type', 'normal')
        
        # Show node ID and type for debugging
        print(f"\n{Fore.BLUE}[Node ID: {node_id}, Type: {node_type}]{Style.RESET_ALL}")
        
        # Format based on speaker
        if speaker == 'Player':
            speaker_format = f"{Fore.CYAN}{speaker}{Style.RESET_ALL}"
        else:
            speaker_format = f"{Fore.YELLOW}{speaker}{Style.RESET_ALL}"
        
        # Display the dialog
        if text:
            print(f"\n{speaker_format}: {text}")
        
        # Display context if present (for debug purposes)
        context = node_data.get('context', '')
        if context and context.strip():
            print(f"{Fore.GREEN}Context: {context}{Style.RESET_ALL}")
        
        # Display jump information if present
        if node_type == 'jump' and node_data.get('goto'):
            print(f"{Fore.YELLOW}[Jump node: Will jump to node {node_data.get('goto')}]{Style.RESET_ALL}")
        # Otherwise display goto if present
        elif node_data.get('goto'):
            if not node_data.get('children'):
                print(f"{Fore.MAGENTA}[Goto: {node_data.get('goto')} (will follow - no children present)]{Style.RESET_ALL}")
            else:
                print(f"{Fore.MAGENTA}[Goto: {node_data.get('goto')} (informational only - children present)]{Style.RESET_ALL}")
        # Display link if present
        if node_data.get('link'):
            if not node_data.get('children') and not node_data.get('goto'):
                print(f"{Fore.MAGENTA}[Link: {node_data.get('link')} (will follow - no children or goto present)]{Style.RESET_ALL}")
            else:
                print(f"{Fore.MAGENTA}[Link: {node_data.get('link')} (informational only)]{Style.RESET_ALL}")
        
        # Display if this is an end node
        if node_data.get('is_end', False):
            print(f"{Fore.RED}[End Node]{Style.RESET_ALL}")
        
        # Display rolls if present
        rolls = node_data.get('rolls', '')
        if rolls and rolls.strip():
            print(f"{Fore.MAGENTA}[Requires roll: {rolls}]{Style.RESET_ALL}")
        
        # Display approval changes
        approvals = node_data.get('approval', [])
        if approvals:
            print(f"{Fore.BLUE}[Companion reactions: {', '.join(approvals)}]{Style.RESET_ALL}")
    
    def get_available_options(self, node_data, ignore_flags=False):
        """Get available dialog options from a node's direct children.

        If ignore_flags is True, include children regardless of flag checks.
        """
        children = node_data.get('children', {})
        
        # Include all direct child nodes, not just ones with text
        meaningful_options = {}
        for child_id, child_data in children.items():
            child_node = self._get_node(child_id)
            if not child_node:
                continue
                
            # Include any child node that meets flag requirements, even if it has no text
            # (such as jump nodes which might not have text)
            if ignore_flags or self._check_flags(child_node):  # Check if flags requirements are met
                meaningful_options[child_id] = child_node
        
        return meaningful_options
    
    def present_options(self, options):
        """Display dialog options with numbered choices"""
        if not options:
            print(f"\n{Fore.RED}[End of dialog - No options available]{Style.RESET_ALL}")
            return None
        
        print(f"\n{Fore.WHITE}Choose your response:{Style.RESET_ALL}")
        option_list = list(options.items())
        
        for i, (option_id, option_data) in enumerate(option_list, 1):
            speaker = option_data.get('speaker', 'Player')
            text = option_data.get('text', '')
            node_type = option_data.get('node_type', 'normal')
            
            # Add visual indicators for options that might have special effects
            indicators = []
            if option_data.get('approval'):
                indicators.append(f"{Fore.BLUE}[Approval]{Style.RESET_ALL}")
            if option_data.get('setflags'):
                indicators.append(f"{Fore.GREEN}[Sets or Removes Flag]{Style.RESET_ALL}")
            if option_data.get('is_end', False):
                indicators.append(f"{Fore.RED}[Ends Dialog]{Style.RESET_ALL}")
            
            # Add jump node indicator
            if node_type == 'jump' and option_data.get('goto'):
                indicators.append(f"{Fore.YELLOW}[Will jump to node {option_data.get('goto')}]{Style.RESET_ALL}")
            # Otherwise add goto indicator
            elif option_data.get('goto'):
                has_children = bool(option_data.get('children', {}))
                if has_children:
                    indicators.append(f"{Fore.MAGENTA}[Info - Goto: {option_data.get('goto')} (not followed - has children)]{Style.RESET_ALL}")
                else:
                    indicators.append(f"{Fore.MAGENTA}[Info - Goto: {option_data.get('goto')} (will follow if chosen - no children)]{Style.RESET_ALL}")
            # Add link indicator if present
            if option_data.get('link'):
                has_children = bool(option_data.get('children', {}))
                has_goto = bool(option_data.get('goto', ''))
                if not has_children and not has_goto:
                    indicators.append(f"{Fore.MAGENTA}[Link: {option_data.get('link')} (will follow if chosen - no children or goto)]{Style.RESET_ALL}")
                else:
                    indicators.append(f"{Fore.MAGENTA}[Link: {option_data.get('link')} (informational only)]{Style.RESET_ALL}")
            
            indicator_text = " ".join(indicators)
            
            # Only show options that have text
            if text:
                print(f"{i}. [{option_id}] {speaker}: {text} {indicator_text}")
            else:
                # For jump nodes without text, show them as jump choices
                if node_type == 'jump':
                    print(f"{i}. [{option_id}] {Fore.YELLOW}[Jump to node {option_data.get('goto')}]{Style.RESET_ALL} {indicator_text}")
                else:
                    # For other options without text, still show them as choices
                    print(f"{i}. [{option_id}] {Fore.CYAN}[Node without text]{Style.RESET_ALL} {indicator_text}")
        
        # Add option to go back to root nodes
        print(f"0. {Fore.RED}[Return to start]{Style.RESET_ALL}")
        
        choice = None
        while choice is None:
            try:
                choice_input = input("\nEnter choice: ")
                choice_num = int(choice_input)
                
                if choice_num == 0:
                    return "START"
                
                if 1 <= choice_num <= len(option_list):
                    choice = option_list[choice_num - 1][0]  # Get the node ID
                else:
                    print(f"{Fore.RED}Invalid choice. Try again.{Style.RESET_ALL}")
                    choice = None
            except ValueError:
                print(f"{Fore.RED}Please enter a number.{Style.RESET_ALL}")
        
        return choice
    
    def show_root_node_selection(self):
        """Show selection menu for root nodes"""
        print(f"\n{Fore.WHITE}===== ROOT NODES ====={Style.RESET_ALL}")
        root_node_list = list(self.root_nodes.items())
        
        for i, (node_id, node_data) in enumerate(root_node_list, 1):
            speaker = node_data.get('speaker', 'Unknown')
            text_preview = node_data.get('text', '')[:50]
            if text_preview:
                text_preview += "..." if len(node_data.get('text', '')) > 50 else ""
                print(f"{i}. [{node_id}] {speaker}: {text_preview}")
            else:
                print(f"{i}. [{node_id}] {speaker}")
        
        print(f"0. {Fore.RED}[Exit simulator]{Style.RESET_ALL}")
        
        choice = None
        while choice is None:
            try:
                choice_input = input("\nSelect a root node to start dialog: ")
                choice_num = int(choice_input)
                
                if choice_num == 0:
                    return None
                
                if 1 <= choice_num <= len(root_node_list):
                    choice = root_node_list[choice_num - 1][0]  # Get the node ID
                else:
                    print(f"{Fore.RED}Invalid choice. Try again.{Style.RESET_ALL}")
                    choice = None
            except ValueError:
                print(f"{Fore.RED}Please enter a number.{Style.RESET_ALL}")
        
        return choice
    
    def follow_node_path(self, node_id, _visited=None, max_hops=200):
        """Follow a path from a node, always following jump nodes and goto for nodes without children.

        Adds cycle protection via a visited set and a hop limit to avoid infinite recursion.
        """
        if _visited is None:
            _visited = set()

        # Detect cycles or excessive hops
        if node_id in _visited:
            print(f"{Fore.YELLOW}[Cycle detected while following path at node {node_id}; stopping follow]{Style.RESET_ALL}")
            return node_id, self._get_node(node_id)
        if len(_visited) >= max_hops:
            print(f"{Fore.YELLOW}[Max follow hops ({max_hops}) reached at node {node_id}; stopping follow]{Style.RESET_ALL}")
            return node_id, self._get_node(node_id)

        _visited.add(node_id)

        node = self._get_node(node_id)
        
        if not node:
            return node_id, None
        
        # ALWAYS follow jump nodes (regardless of whether they have children)
        node_type = node.get('node_type', 'normal')
        if node_type == 'jump' and node.get('goto'):
            goto_node_id = node.get('goto')
            if goto_node_id:
                print(f"{Fore.MAGENTA}[Following jump node link to node {goto_node_id}]{Style.RESET_ALL}")
                return self.follow_node_path(goto_node_id, _visited, max_hops)  # Recursively follow jump/goto chains
        
        # For non-jump nodes, only follow goto if they have no children
        elif not node.get('children') and node.get('goto'):
            goto_node_id = node.get('goto')
            if goto_node_id:
                print(f"{Fore.MAGENTA}[Following goto link to node {goto_node_id} (no children present)]{Style.RESET_ALL}")
                return self.follow_node_path(goto_node_id, _visited, max_hops)  # Recursively follow goto chains
        
        # For nodes with no children and no goto, follow link if present
        elif not node.get('children') and not node.get('goto') and node.get('link'):
            link_node_id = node.get('link')
            if link_node_id:
                print(f"{Fore.MAGENTA}[Following link to node {link_node_id} (no children or goto present)]{Style.RESET_ALL}")
                return self.follow_node_path(link_node_id, _visited, max_hops)  # Recursively follow link chains
            
        # Otherwise, return the node directly
        return node_id, node
    
    def interactive_mode(self):
        """Start the interactive dialog mode"""
        print(f"\n{Fore.WHITE}===== DIALOG SIMULATOR - INTERACTIVE MODE ====={Style.RESET_ALL}")
        print("Explore the dialog tree by selecting options.")
        
        while True:
            # Show root node selection
            root_node_id = self.show_root_node_selection()
            if not root_node_id:
                break
            
            self.explore_dialog_from_node(root_node_id)
            
            # Show companion approval status
            self.show_companion_status()
    
    def explore_dialog_from_node(self, start_node_id, export_txt=False, export_json=False, export_approval=False):
        """Explore dialog starting from a specific node, always following jump nodes and traversing child nodes
        
        Args:
            start_node_id (str): The node ID to start from
            export_txt (bool): Whether to export the traversal to a text file
            export_json (bool): Whether to export the traversal to a JSON file
            export_approval (bool): Whether to export approval history to a JSON file
            
        Returns:
            tuple: (visited_nodes, txt_file_path, json_file_path, approval_file_path)
        """
        current_node_id = start_node_id
        self.visited_nodes = []
        
        # For testing purposes, capture the original active flags to restore later
        original_flags = self.active_flags.copy()
        test_mode = True  # Flag to indicate we're in test mode and should ignore flag requirements
        
        print(f"{Fore.CYAN}Starting dialog from node {current_node_id}{Style.RESET_ALL}")
        
        # Keep track of full details for each visited node
        visited_node_details = []
        
        while current_node_id:
            # First, follow any jump nodes or goto links as needed
            original_node_id = current_node_id
            current_node_id, current_node = self.follow_node_path(current_node_id)
            
            if original_node_id != current_node_id:
                if self._get_node(original_node_id).get('node_type') == 'jump':
                    print(f"{Fore.CYAN}Followed jump node from {original_node_id} to {current_node_id}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.CYAN}Followed path from node {original_node_id} to {current_node_id}{Style.RESET_ALL}")
            
            if not current_node:
                print(f"{Fore.RED}Error: Node {current_node_id} not found{Style.RESET_ALL}")
                break
            
            # Add to visited nodes
            self.visited_nodes.append(current_node_id)
            
            # Also store node data for export
            if current_node:
                # Create a simplified copy of the node data without mutating original node
                node_data = {
                    "id": current_node_id,
                    "speaker": current_node.get('speaker', 'Unknown'),
                    "text": self._get_effective_text(current_node),
                    "node_type": current_node.get('node_type', 'normal'),
                    "checkflags": current_node.get('checkflags', []),
                    "setflags": current_node.get('setflags', []),
                    "goto": current_node.get('goto', ''),
                    "link": current_node.get('link', ''),
                    "is_end": current_node.get('is_end', False),
                    "approval": current_node.get('approval', []),
                    "context": current_node.get('context', '')
                }
                visited_node_details.append(node_data)
            
            # Display the current node
            self.display_node(current_node_id, current_node)
            
            # Process approvals and flags
            self._process_approvals(current_node)
            self._process_setflags(current_node)
            
            # Check if this is an end node explicitly marked as is_end
            if current_node.get('is_end', False):
                print(f"\n{Fore.RED}[End of dialog path - Explicit end node]{Style.RESET_ALL}")
                break
            
            # Get available options based on this node's direct children
            # In test mode, ignore flag requirements when showing options
            options = self.get_available_options(current_node, ignore_flags=test_mode)
                
            # If there are no options, end the dialog
            if not options:
                print(f"\n{Fore.RED}[End of dialog path - No more options]{Style.RESET_ALL}")
                break
                
            # Present options to the user
            choice = self.present_options(options)
            
            if choice == "START":
                break
            elif choice:
                current_node_id = choice
            else:
                # No valid choice returned
                print(f"\n{Fore.RED}[Dialog ended]{Style.RESET_ALL}")
                break
                
        print(f"\n{Fore.CYAN}[Dialog sequence complete - Visited {len(self.visited_nodes)} nodes]{Style.RESET_ALL}")
        
        # Restore original flags if in test mode
        if test_mode:
            self.active_flags = original_flags
            
        # Export results if requested
        txt_file = None
        json_file = None
        approval_file = None
        
        if export_txt and self.visited_nodes:
            # Export to text file
            output_file = f'node_{start_node_id}_traversal.txt'
            print(f"{Fore.GREEN}Exporting traversal to {output_file}...{Style.RESET_ALL}")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Dialog Traversal from Node {start_node_id}\n")
                f.write(f"Total nodes visited: {len(self.visited_nodes)}\n\n")
                f.write(f"Path: {' -> '.join(self.visited_nodes)}\n\n")
                
                # Add details for each node in the path using our custom format
                f.write(f"Detailed Traversal:\n")
                for node_data in visited_node_details:
                    # Skip nodes without text
                    if not node_data['text']:
                        continue
                        
                    # Format the line according to requirements
                    line = ""
                    
                    # Handle speaker/text part based on node type
                    if node_data['node_type'] == 'tagcinematic':
                        line = f"[description] {node_data['text']}"
                    else:
                        line = f"{node_data['speaker']}: {node_data['text']}"
                    
                    # Add context if present (context isn't captured in node_data by default, so would need to be added)
                    context = node_data['context']
                    if context:
                        line += f" || [context] {context}"
                    
                    # Add approval changes if present
                    if node_data['approval']:
                        line += f" || [approval] {', '.join(node_data['approval'])}"
                    if ": true" in line.lower() or ": false" in line.lower():
                        continue
                    # Write the formatted line
                    f.write(f"{line}\n")
            
            txt_file = output_file
            print(f"{Fore.GREEN}Traversal exported to {txt_file}{Style.RESET_ALL}")
        
        if export_json and self.visited_nodes:
            # Export to JSON file
            output_file = f'node_{start_node_id}_traversal.json'
            print(f"{Fore.GREEN}Exporting traversal data to {output_file}...{Style.RESET_ALL}")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "start_node": start_node_id,
                    "path": self.visited_nodes,
                    "nodes": visited_node_details
                }, f, indent=2, ensure_ascii=False)
            
            json_file = output_file
            print(f"{Fore.GREEN}Traversal data exported to {json_file}{Style.RESET_ALL}")
        
        # Export approval history if requested
        if export_approval and any(len(changes) > 0 for changes in self.companion_approval_history.values()):
            approval_file = f'node_{start_node_id}_approvals.json'
            self.export_approval_history(approval_file)
        
        return self.visited_nodes, txt_file, json_file, approval_file
    
    def _is_leaf_node(self, node_id):
        """Check if a node is a leaf node (end of dialog)"""
        node = self._get_node(node_id)
        if not node:
            return False
            
        # A node is a leaf if it's explicitly marked as an end node
        if node.get('is_end', False):
            return True
            
        # A node is NOT a leaf if it has a goto link
        if node.get('goto'):
            return False
            
        # A node is NOT a leaf if it has a link field
        if node.get('link'):
            return False
            
        # A node is a leaf if it has no valid children
        children = self.get_available_options(node)
        if not children:
            return True
            
        return False
    
    def _simulate_paths_from_node(self, node_id, current_path, depth, max_depth, test_mode=False, verbose=False, deadline=None, max_paths=None, paths_generated=None):
        """Recursively simulate all paths from a node, always following jump nodes and goto for nodes without children

        Supports optional limits via:
          - deadline (float): monotonic() timestamp after which traversal should stop, yielding TIMEOUT_REACHED marker
          - max_paths (int): maximum number of leaf paths to emit for this traversal/root
          - paths_generated (dict): mutable counter with key 'count' tracking emitted leaf paths
        """
        # Global time cutoff
        if deadline is not None and time.monotonic() > deadline:
            return [current_path + [node_id] + ["TIMEOUT_REACHED"]]

        # Prevent excessive recursion depth
        if depth >= max_depth:
            if paths_generated is not None and (max_paths is None or paths_generated.get('count', 0) < max_paths):
                paths_generated['count'] = paths_generated.get('count', 0) + 1
                return [current_path + [node_id] + ["MAX_DEPTH_REACHED"]]
            return []
        
        node = self._get_node(node_id)
        if not node:
            if paths_generated is not None and (max_paths is None or paths_generated.get('count', 0) < max_paths):
                paths_generated['count'] = paths_generated.get('count', 0) + 1
                return [current_path + [node_id] + ["NODE_NOT_FOUND"]]
            return []
        
        # Add current node to path
        current_path = current_path + [node_id]
        
        # ALWAYS follow jump nodes first
        node_type = node.get('node_type', 'normal')
        if node_type == 'jump' and node.get('goto'):
            goto_id = node.get('goto')
            if verbose:
                print(f"{Fore.MAGENTA}  [During simulation: Node {node_id} is a jump node, jumping to {goto_id}]{Style.RESET_ALL}")
            # Increment depth to avoid infinite recursion through jump chains
            return self._simulate_paths_from_node(goto_id, current_path, depth + 1, max_depth, test_mode, verbose, deadline, max_paths, paths_generated)
        
        # For nodes with goto but no children, follow the goto
        if not node.get('children') and node.get('goto'):
            goto_id = node.get('goto')
            if verbose:
                print(f"{Fore.MAGENTA}  [During simulation: Node {node_id} has no children, following goto to {goto_id}]{Style.RESET_ALL}")
            # Increment depth to ensure max_depth applies across goto chains
            return self._simulate_paths_from_node(goto_id, current_path, depth + 1, max_depth, test_mode, verbose, deadline, max_paths, paths_generated)
            
        # For nodes with link but no children and no goto, follow the link
        if not node.get('children') and not node.get('goto') and node.get('link'):
            link_id = node.get('link')
            if verbose:
                print(f"{Fore.MAGENTA}  [During simulation: Node {node_id} has no children or goto, following link to {link_id}]{Style.RESET_ALL}")
            # Increment depth to ensure link chains are bounded by max_depth
            return self._simulate_paths_from_node(link_id, current_path, depth + 1, max_depth, test_mode, verbose, deadline, max_paths, paths_generated)
        
        # Check if we've reached a leaf node (that has no goto or link)
        # Since we've already checked for goto and link above, if it's a leaf node here, it truly is an end
        if self._is_leaf_node(node_id) and not test_mode:
            if verbose:
                print(f"{Fore.RED}  [During simulation: Node {node_id} is a true leaf node]{Style.RESET_ALL}")
            if paths_generated is not None and (max_paths is None or paths_generated.get('count', 0) < max_paths):
                paths_generated['count'] = paths_generated.get('count', 0) + 1
                return [current_path]
            return []
        
        # Get all available options based on direct children
        children = self.get_available_options(node, ignore_flags=test_mode)
            
        if not children:
            # This is a leaf node with no options, no goto, and no link (already checked above)
            if verbose:
                print(f"{Fore.RED}  [During simulation: Node {node_id} has no children, goto, or link - ending path]{Style.RESET_ALL}")
            if paths_generated is not None and (max_paths is None or paths_generated.get('count', 0) < max_paths):
                paths_generated['count'] = paths_generated.get('count', 0) + 1
                return [current_path]
            return []
        
        # Explore all child paths
        all_paths = []
        for child_id in children:
            # Check time limit before diving into each child
            if deadline is not None and time.monotonic() > deadline:
                all_paths.append(current_path + [node_id] + ["TIMEOUT_REACHED"])
                break
            child_paths = self._simulate_paths_from_node(child_id, current_path, depth + 1, max_depth, test_mode, verbose, deadline, max_paths, paths_generated)
            all_paths.extend(child_paths)
            # Enforce max paths limit if provided
            if paths_generated is not None and max_paths is not None and paths_generated.get('count', 0) >= max_paths:
                break
        
        # If no children produced paths (shouldn't happen), return current path
        if not all_paths:
            if paths_generated is not None and (max_paths is None or paths_generated.get('count', 0) < max_paths):
                paths_generated['count'] = paths_generated.get('count', 0) + 1
                return [current_path]
            return []
            
        return all_paths
    
    def simulate_all_paths(self, max_depth=20, print_paths=True, test_mode=False, export_txt=False, export_json=False, export_dict=False, verbose=False, sample_roots=False, max_roots=5, only_longest=False, time_limit_seconds=None, max_paths_per_root=None):
        """Simulate all possible dialog paths for each root node
        
        Args:
            max_depth (int): Maximum depth to traverse
            print_paths (bool): Whether to print paths to console
            test_mode (bool): Whether to ignore flag requirements
            export_txt (bool): Whether to export paths to a text file
            export_json (bool): Whether to export traversals to a JSON file
            export_dict (bool): Whether to export paths to a Python dictionary file
            verbose (bool): Whether to print detailed simulation logs
            
        Returns:
            tuple: (all_paths, txt_file_path, json_file_path, dict_file_path)
        """
        print(f"\n{Fore.WHITE}===== DIALOG SIMULATOR - SIMULATION MODE ====={Style.RESET_ALL}")
        # show metadata
        self.display_metadata()
        print(f"Simulating all dialog paths (max depth {max_depth} if no leaf node found)...")
        
        if test_mode:
            print(f"{Fore.YELLOW}Running in TEST MODE - Flag requirements will be ignored{Style.RESET_ALL}")
        
        if verbose:
            print(f"{Fore.BLUE}Verbose mode enabled - Detailed simulation logs will be shown{Style.RESET_ALL}")
        
        # Store original flags to restore later if in test mode
        original_flags = self.active_flags.copy() if test_mode else None
        
        all_paths = []
        total_leaf_paths = 0
        
        # Determine which root nodes to traverse
        root_nodes_list = list(self.root_nodes.items())
        if sample_roots and root_nodes_list:
            sample_count = min(max_roots, len(root_nodes_list))
            roots_to_traverse = random.sample(root_nodes_list, sample_count)
        else:
            roots_to_traverse = root_nodes_list

        for root_id, root_data in roots_to_traverse:
            print(f"\n{Fore.YELLOW}Root Node: {root_id} - {root_data.get('speaker', 'Unknown')}{Style.RESET_ALL}")
            deadline = (time.monotonic() + time_limit_seconds) if time_limit_seconds else None
            paths_counter = {'count': 0}
            paths = self._simulate_paths_from_node(root_id, [], 0, max_depth, test_mode, verbose, deadline=deadline, max_paths=max_paths_per_root, paths_generated=paths_counter)
            
            # Optionally select only the longest path for this root node
            if only_longest and paths:
                longest_path = max(paths, key=len)
                paths = [longest_path]
            
            # Count how many of these paths ended at true leaf nodes
            leaf_paths = [p for p in paths if self._is_leaf_node(p[-1])]
            total_leaf_paths += len(leaf_paths)
            
            if print_paths:
                for i, path in enumerate(paths, 1):
                    is_leaf = self._is_leaf_node(path[-1])
                    leaf_marker = f"{Fore.GREEN}[LEAF NODE]{Style.RESET_ALL}" if is_leaf else ""
                    
                    # Add goto or link marker if the last node has one
                    follow_info = ""
                    last_node_id = path[-1]
                    if last_node_id not in ["MAX_DEPTH_REACHED", "NODE_NOT_FOUND"]:
                        last_node = self._get_node(last_node_id)
                        if last_node:
                            if last_node.get('goto'):
                                follow_info = f"{Fore.MAGENTA} [GOTO: {last_node.get('goto')}]{Style.RESET_ALL}"
                            elif last_node.get('link'):
                                follow_info = f"{Fore.MAGENTA} [LINK: {last_node.get('link')}]{Style.RESET_ALL}"
                    
                    print(f"\nPath {i}: {' -> '.join(path)} {leaf_marker}{follow_info}")
            
            print(f"Total paths from root {root_id}: {len(paths)}")
            if any(p and p[-1] == "TIMEOUT_REACHED" for p in paths):
                print(f"{Fore.YELLOW}Time limit reached for root {root_id}{Style.RESET_ALL}")
            print(f"Paths ending at leaf nodes: {len(leaf_paths)}")
            all_paths.extend(paths)
        
        print(f"\nTotal dialog paths: {len(all_paths)}")
        print(f"Total paths ending at leaf nodes: {total_leaf_paths}")
        
        # Restore original flags if in test mode
        if test_mode and original_flags is not None:
            self.active_flags = original_flags
            
        # Export results if requested
        txt_file = None
        json_file = None
        dict_file = None
        
        if export_txt:
            txt_file = self.export_paths_to_txt(all_paths)
            
        if export_json:
            # Create structured traversal data
            traversals = self.create_traversal_data(all_paths)
            json_file = self.export_traversals_to_json(traversals)
            
        if export_dict:
            dict_file = self.export_paths_to_dict(all_paths)
            
        return all_paths, txt_file, json_file, dict_file

    def _path_contains_approval(self, path):
        """Return True if any node in the path has a non-empty approval list."""
        for node_id in path:
            if node_id in ["MAX_DEPTH_REACHED", "NODE_NOT_FOUND"]:
                continue
            node = self._get_node(node_id)
            if not node:
                continue
            approvals = node.get('approval', [])
            if approvals:
                return True
        return False

    def simulate_approval_paths(self, max_depth=20, print_paths=True, test_mode=False, export_txt=False, export_json=False, export_dict=False, verbose=False, time_limit_seconds=None, max_paths_per_root=None):
        """Simulate only the dialog paths that include at least one node with non-empty approvals.
        
        Args:
            max_depth (int): Maximum depth to traverse
            print_paths (bool): Whether to print paths to console
            test_mode (bool): Whether to ignore flag requirements
            export_txt (bool): Whether to export paths to a text file
            export_json (bool): Whether to export traversals to a JSON file
            export_dict (bool): Whether to export paths to a Python dictionary file
            verbose (bool): Whether to print detailed simulation logs
        
        Returns:
            tuple: (approval_paths, txt_file_path, json_file_path, dict_file_path)
        """
        print(f"\n{Fore.WHITE}===== DIALOG SIMULATOR - APPROVAL PATHS ====={Style.RESET_ALL}")
        self.display_metadata()
        print(f"Simulating only dialog paths that include approvals (max depth {max_depth} if no leaf node found)...")
        
        if test_mode:
            print(f"{Fore.YELLOW}Running in TEST MODE - Flag requirements will be ignored{Style.RESET_ALL}")
        if verbose:
            print(f"{Fore.BLUE}Verbose mode enabled - Detailed simulation logs will be shown{Style.RESET_ALL}")
        
        original_flags = self.active_flags.copy() if test_mode else None
        
        approval_paths = []
        total_leaf_paths = 0
        
        # Explore all root nodes (no sampling) to avoid missing approval paths
        for root_id, root_data in self.root_nodes.items():
            print(f"\n{Fore.YELLOW}Root Node: {root_id} - {root_data.get('speaker', 'Unknown')}{Style.RESET_ALL}")
            deadline = (time.monotonic() + time_limit_seconds) if time_limit_seconds else None
            paths_counter = {'count': 0}
            paths = self._simulate_paths_from_node(root_id, [], 0, max_depth, test_mode, verbose, deadline=deadline, max_paths=max_paths_per_root, paths_generated=paths_counter)
            paths_with_approvals = [p for p in paths if self._path_contains_approval(p)]
            
            leaf_paths = [p for p in paths_with_approvals if self._is_leaf_node(p[-1])]
            total_leaf_paths += len(leaf_paths)
            
            if print_paths:
                for i, path in enumerate(paths_with_approvals, 1):
                    is_leaf = self._is_leaf_node(path[-1])
                    leaf_marker = f"{Fore.GREEN}[LEAF NODE]{Style.RESET_ALL}" if is_leaf else ""
                    follow_info = ""
                    last_node_id = path[-1]
                    if last_node_id not in ["MAX_DEPTH_REACHED", "NODE_NOT_FOUND"]:
                        last_node = self._get_node(last_node_id)
                        if last_node:
                            if last_node.get('goto'):
                                follow_info = f"{Fore.MAGENTA} [GOTO: {last_node.get('goto')}]" + f"{Style.RESET_ALL}"
                            elif last_node.get('link'):
                                follow_info = f"{Fore.MAGENTA} [LINK: {last_node.get('link')}]" + f"{Style.RESET_ALL}"
                    print(f"\nPath {i}: {' -> '.join(path)} {leaf_marker}{follow_info}")
            
            print(f"Total approval paths from root {root_id}: {len(paths_with_approvals)}")
            if any(p and p[-1] == "TIMEOUT_REACHED" for p in paths):
                print(f"{Fore.YELLOW}Time limit reached for root {root_id}{Style.RESET_ALL}")
            print(f"Paths ending at leaf nodes: {len(leaf_paths)}")
            approval_paths.extend(paths_with_approvals)
        
        print(f"\nTotal approval-containing dialog paths: {len(approval_paths)}")
        print(f"Total approval paths ending at leaf nodes: {total_leaf_paths}")
        
        if test_mode and original_flags is not None:
            self.active_flags = original_flags
        
        txt_file = None
        json_file = None
        dict_file = None
        
        if export_txt:
            txt_file = self.export_paths_to_txt(approval_paths)
        if export_json:
            traversals = self.create_traversal_data(approval_paths)
            json_file = self.export_traversals_to_json(traversals)
        if export_dict:
            dict_file = self.export_paths_to_dict(approval_paths)
        
        return approval_paths, txt_file, json_file, dict_file

    def simulate_approval_paths_with_qa(self, max_depth=20, test_mode=False, relevant_flag_prefixes=None, export_qa_path=None, verbose=False):
        """Simulate approval paths and build QA examples that include relevant flag context.

        Args:
            max_depth (int): traversal bound
            test_mode (bool): bypass flag requirements during traversal
            relevant_flag_prefixes (list[str]|None): prefixes to filter flags
            export_qa_path (str|None): if provided, export QA dataset to this JSON
            verbose (bool): detailed logs

        Returns:
            list: QA example dicts
        """
        approval_paths, _, _, _ = self.simulate_approval_paths(
            max_depth=max_depth,
            print_paths=False,
            test_mode=test_mode,
            export_txt=False,
            export_json=False,
            export_dict=False,
            verbose=verbose,
        )

        qa_examples = []
        for path in approval_paths:
            # Build QA examples per path; use default flags as initial context baseline
            examples = self.build_qa_examples_from_path(
                path,
                initial_flags=self.default_flags,
                relevant_flag_prefixes=relevant_flag_prefixes,
            )
            qa_examples.extend(examples)

        if export_qa_path:
            self.export_qa_dataset(qa_examples, export_qa_path)

        return qa_examples

    def _find_cluster_for_path(self, cluster_index_file='goals_to_json_paths.json'):
        """Return (cluster_key, file_list, current_index) for self.json_path if present, else (None, None, None)."""
        if not os.path.isfile(cluster_index_file):
            return None, None, None
        try:
            with open(cluster_index_file, 'r', encoding='utf-8') as f:
                clusters = json.load(f)
        except Exception:
            return None, None, None

        # Normalize for comparisons
        current_abs = os.path.abspath(self.json_path)
        for key, paths in clusters.items():
            for idx, p in enumerate(paths):
                cand_abs = os.path.abspath(p)
                if cand_abs == current_abs:
                    return key, paths, idx
        return None, None, None

    def _gather_cluster_synopses(self, cluster_index_file='goals_to_json_paths.json', include_current=True, include_prior=True, include_future=False, limit=10, use_context_outputs=False, contexts_root='qa-contexts'):
        """Collect synopses or prior generated contexts from the same cluster as the current file.

        Returns list of tuples (path, text), capped by limit while preserving order.
        If use_context_outputs=True, attempts to load '<contexts_root>/<rel>/<stem>_context.txt'
        instead of reading the raw JSON synopsis.
        """
        cluster_key, cluster_files, current_idx = self._find_cluster_for_path(cluster_index_file)
        if cluster_files is None or current_idx is None:
            return []

        def _act_key_from_path(path):
            # Produce a monotonically increasing key reflecting Act ordering
            # Examples: Act1 < Act2 < Act2b < Act3 < Act3i
            try:
                parts = os.path.normpath(path).split(os.sep)
                token = None
                for seg in parts:
                    low = seg.lower()
                    if low.startswith('act'):
                        token = seg
                        break
                    if low == 'prologue':
                        token = 'Prologue'
                        break
                if token is None:
                    return 1_000_000  # Unknown, treat as very late
                lowtok = token.lower()
                if lowtok == 'prologue':
                    return 0
                # Parse pattern Act<number><suffix?>
                # Default: suffix empty -> 0, letters map: a->1, b->2, ..., i->9
                prefix = 'act'
                num_part = ''
                suf_part = ''
                for ch in token[len(prefix):]:
                    if ch.isdigit():
                        num_part += ch
                    else:
                        suf_part += ch
                base = int(num_part) if num_part else 999
                suf_rank = 0
                if suf_part:
                    first = suf_part[0].lower()
                    if 'a' <= first <= 'z':
                        suf_rank = ord(first) - ord('a') + 1
                return base * 10 + suf_rank
            except Exception:
                return 1_000_000

        current_act_key = _act_key_from_path(self.json_path)

        selected_paths = []
        if include_prior and current_idx > 0:
            selected_paths.extend(cluster_files[:current_idx])
        if include_current:
            selected_paths.append(cluster_files[current_idx])
        if include_future and current_idx + 1 < len(cluster_files):
            selected_paths.extend(cluster_files[current_idx + 1:])

        # Filter out paths from later Acts than the current file
        selected_paths = [p for p in selected_paths if _act_key_from_path(p) <= current_act_key]

        # Helper to map raw JSON path to generated context path
        def _context_path_for_json(jpath):
            norm = os.path.normpath(jpath)
            # Derive relative path under 'output/' if present
            rel = None
            output_prefix = 'output' + os.sep
            if norm.startswith(output_prefix):
                rel = norm[len(output_prefix):]
            else:
                marker = os.sep + 'output' + os.sep
                idx = norm.find(marker)
                if idx != -1:
                    rel = norm[idx + len(marker):]
            if rel is None:
                rel = os.path.basename(norm)
            rel_dir = os.path.dirname(rel)
            stem = os.path.splitext(os.path.basename(rel))[0]
            return os.path.join(contexts_root, rel_dir, f"{stem}_context.txt")

        # Cap and read synopses or contexts
        syns = []
        for p in selected_paths[:limit]:
            if use_context_outputs:
                ctx_path = _context_path_for_json(p)
                try:
                    with open(ctx_path, 'r', encoding='utf-8') as fctx:
                        text = fctx.read().strip()
                        if text:
                            syns.append((p, text))
                            continue
                except Exception:
                    # Fall through to attempt raw synopsis if context missing
                    pass
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    syn = data.get('metadata', {}).get('synopsis', '')
                    if syn:
                        syns.append((p, syn))
            except Exception:
                continue
        return syns

    def _collect_current_file_canonical(self, relevant_flag_prefixes=None, max_depth=25, test_mode=True):
        """Collect canonical turns, top-3 sample windows, and aggregated flags for the current file."""
        approval_paths, _, _, _ = self.simulate_approval_paths(
            max_depth=max_depth,
            print_paths=False,
            test_mode=test_mode,
            export_txt=False,
            export_json=False,
            export_dict=False,
            verbose=False,
            time_limit_seconds=120,
        )

        all_examples = []
        for path in approval_paths:
            exs = self.build_qa_examples_from_path(
                path,
                initial_flags=self.default_flags,
                relevant_flag_prefixes=relevant_flag_prefixes
            )
            all_examples.extend(exs)

        if not all_examples:
            # Fallback: derive samples from general simulated paths (not limited to approvals)
            all_paths, _, _, _ = self.simulate_all_paths(
                max_depth=max_depth,
                print_paths=False,
                test_mode=test_mode,
                export_txt=False,
                export_json=False,
                export_dict=False,
                verbose=False,
                only_longest=False,
                time_limit_seconds=2,
                max_paths_per_root=5
            )

            if not all_paths:
                return [], [], []

            # Pick top-3 longest paths and convert to dialogue windows
            sorted_paths = sorted(all_paths, key=lambda p: len(p), reverse=True)[:3]
            def _path_to_turns(path_ids):
                turns = []
                for nid in path_ids:
                    if nid in ["MAX_DEPTH_REACHED", "NODE_NOT_FOUND", "TIMEOUT_REACHED"]:
                        continue
                    node = self._get_node(nid)
                    if not node:
                        continue
                    turns.append({
                        'node_id': nid,
                        'speaker': node.get('speaker', 'Unknown'),
                        'text': self._get_effective_text(node),
                        'context': node.get('context', ''),
                    })
                return turns

            fallback_examples = []
            for spath in sorted_paths:
                turns = _path_to_turns(spath)
                if turns:
                    fallback_examples.append({
                        'approval_node_id': turns[-1]['node_id'],
                        'context_dialogue': turns,
                        'approval_list': [],
                        'active_flags': [],
                        'full_active_flags_count': 0,
                    })

            if not fallback_examples:
                return [], [], []

            all_examples = fallback_examples

        canonical = max(all_examples, key=lambda e: len(e.get('context_dialogue', [])))
        canonical_turns = canonical.get('context_dialogue', [])

        # Select top-3 longest dialogue windows as representative samples
        top_examples = sorted(
            all_examples,
            key=lambda e: len(e.get('context_dialogue', [])),
            reverse=True
        )[:3]
        top3_samples = [ex.get('context_dialogue', []) for ex in top_examples]

        aggregated_flags = set()
        for ex in all_examples:
            for fl in ex.get('active_flags', []):
                aggregated_flags.add(fl)
        return canonical_turns, sorted(aggregated_flags), top3_samples

    def generate_cluster_context_for_current_file(self, cluster_index_file='goals_to_json_paths.json', model='openai/gpt-5-mini', temperature=0.2, max_tokens=8000, relevant_flag_prefixes=None, max_depth=50, test_mode=True, verbose=False):
        """Generate a single LLM context for this raw file using relevant files in the same cluster.

        Includes:
          - Current file synopsis
          - Prior related dialogue synopses (from cluster index, up to current file)
          - Aggregated relevant flags across this file's approval examples
          - Canonical dialogue excerpt from this file
        """
        # Get canonical data for current file
        canonical_turns, aggregated_flags, top3_samples = self._collect_current_file_canonical(
            relevant_flag_prefixes=relevant_flag_prefixes,
            max_depth=max_depth,
            test_mode=test_mode,
        )

        # Find cluster and collect prior synopses
        cluster_key, cluster_files, current_idx = self._find_cluster_for_path(cluster_index_file)
        prior_synopses = []
        if cluster_files is not None and current_idx is not None:
            prior_files = cluster_files[:current_idx]
            for p in prior_files[-10:]:  # cap to last 10 to keep prompt size reasonable
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        syn = data.get('metadata', {}).get('synopsis', '')
                        if syn:
                            prior_synopses.append((p, syn))
                except Exception:
                    continue

        # Gather cluster contexts/synopses (prefer previously generated contexts, prior-only, Act-aware)
        cluster_synopses = self._gather_cluster_synopses(
            cluster_index_file=cluster_index_file,
            include_current=False,
            include_prior=True,
            include_future=False,
            limit=10,
            use_context_outputs=True,
            contexts_root='qa-contexts'
        )

        current_synopsis = self.metadata.get('synopsis', '')
        dialogue_samples = top3_samples if top3_samples else [canonical_turns]
        # Default: RAG using clusters and cosine similarity over synopses/contexts
        return self.generate_context_with_rag(
            synopsis=current_synopsis,
            dialogue_samples=dialogue_samples,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=5,
            cluster_index_file=cluster_index_file,
            contexts_root='qa-contexts',
            model_name=None,
            restrict_to_current_clusters=True
        )

    def generate_context_with_llm(self, synopsis, dialogue_samples, active_flags=None, model='gpt-5-mini', temperature=0.2, max_tokens=8000, cluster_synopses=None):
        """
        Use an LLM (via LiteLLM) to synthesize necessary background context for QA data
        using the dialogue's synopsis, selected dialogue turns (with text/context),
        and optionally currently active flags as signals.

        Args:
            synopsis (str): Metadata synopsis for the dialogue
            dialogue_samples (list[list[dict]]|list[dict]): Either a list of 3 dialogue samples,
                each being a list of turn dicts, or a single list of turn dicts
            active_flags (iterable[str]|None): Optional flags to provide additional context
            model (str): 'gpt-5' or 'gemini/gemini-2.5-flash' or any LiteLLM model id
            temperature (float): Sampling temperature
            max_tokens (int): Max response tokens
            cluster_synopses (list[tuple[str,str]]|None): Optional list of (path, synopsis) for
                dialogues in the same cluster to include in the prompt.

        Returns:
            str: Generated context paragraph(s), or an empty string if LiteLLM unavailable.
        """
        if not LITELLM_AVAILABLE:
            print(f"{Fore.YELLOW}LiteLLM is not installed. Skipping LLM context generation.{Style.RESET_ALL}")
            return ""

        # Normalize input to a list of dialogue samples
        if dialogue_samples and isinstance(dialogue_samples, list) and dialogue_samples and isinstance(dialogue_samples[0], dict):
            # Caller passed a single flat list of turns; wrap it
            dialogue_samples_norm = [dialogue_samples]
        else:
            dialogue_samples_norm = dialogue_samples or []

        # Prepare a compact prompt
        sample_blocks = []
        for sample_index, sample_turns in enumerate(dialogue_samples_norm, start=1):
            dialogue_lines = []
            for turn in sample_turns:
                speaker = turn.get('speaker', 'Unknown')
                text = turn.get('text', '')
                ctx = turn.get('context', '')
                node_id = turn.get('node_id', '')
                if text:
                    line = f"- {speaker}: {text}"
                    if ctx:
                        line += f" || [context] {ctx}"
                    line += f" (node {node_id})"
                    dialogue_lines.append(line)
                elif ctx:
                    # Fallback: include context-only lines if text is empty
                    line = f"- [context] {ctx} (node {node_id})"
                    dialogue_lines.append(line)
            if dialogue_lines:
                sample_blocks.append(f"Sample {sample_index}:\n" + "\n".join(dialogue_lines))

        prompt_instructions = (
            "You are assisting to build QA datasets from single-dialogue windows in Baldur's Gate 3. "
            "Given a synopsis and up to three example sequences of turns, write a concise background context summary that explains any off-screen state, prior events, or relationships needed to understand the player and NPC lines. "
            "These sequences are only examples (i.e., possible turn-outs), and do not represent all possibilities. Use them as references only; emphasize the synopsis and any provided cluster contexts. "
            "Emphasize facts consistent with the synopsis and cluster contexts. Avoid inventing plot beyond general inferences. Keep it 3-6 sentences."
        )

        # Build optional cluster context block
        cluster_block = ""
        if cluster_synopses:
            try:
                lines = [f"- {os.path.basename(p)}: {syn}" for p, syn in cluster_synopses]
            except Exception:
                lines = [f"- {p}: {syn}" for p, syn in cluster_synopses]
            cluster_block = "Cluster Contexts:\n" + "\n".join(lines) + "\n\n"

        user_content = (
            f"Synopsis:\n{synopsis}\n\n"
            + cluster_block
            + ("Dialogue Excerpts (up to 3 samples, ordered):\n" + "\n\n".join(sample_blocks) if sample_blocks else "No dialogue samples provided.")
        )

        # Record the last LLM input content for downstream metadata writing
        self.last_llm_input = user_content

        # Map friendly names to LiteLLM model ids if needed
        model_id = model
        if 'gpt' in model.lower():
            model_id = f'{model}'
        elif model.lower() in ('gemini-2.5-flash', 'gemini/gemini-2.5-flash'):
            model_id = 'gemini-2.5-flash'
        try:
            resp = litellm_completion(
                model=model_id,
                messages=[
                    {"role": "system", "content": prompt_instructions},
                    {"role": "user", "content": user_content},
                ],
                max_completion_tokens=max_tokens
            )
            
            # LiteLLM standardizes OpenAI-like responses
            return resp.choices[0].message["content"].strip()
        except Exception as e:
            print(f"{Fore.RED}LLM context generation failed: {e}{Style.RESET_ALL}")
            return ""

    def show_companion_status(self):
        """Display current companion approval status"""
        print(f"\n{Fore.CYAN}===== COMPANION APPROVAL STATUS ====={Style.RESET_ALL}")
        for companion, value in self.companion_approvals.items():
            if value > 0:
                status = f"{Fore.GREEN}+{value}{Style.RESET_ALL}"
            elif value < 0:
                status = f"{Fore.RED}{value}{Style.RESET_ALL}"
            else:
                status = f"{Fore.WHITE}{value}{Style.RESET_ALL}"
            
            # Show count of changes in history
            change_count = len(self.companion_approval_history[companion])
            if change_count > 0:
                print(f"{companion}: {status} ({change_count} changes)")
            else:
                print(f"{companion}: {status}")
        
        # Option to show detailed history
        if any(len(changes) > 0 for changes in self.companion_approval_history.values()):
            show_details = input(f"\nShow approval change history? (y/n): ").lower() == 'y'
            if show_details:
                self.show_approval_history()

    def show_approval_history(self):
        """Display the detailed history of approval changes"""
        print(f"\n{Fore.CYAN}===== COMPANION APPROVAL HISTORY ====={Style.RESET_ALL}")
        
        any_changes = False
        for companion, changes in self.companion_approval_history.items():
            if changes:
                any_changes = True
                print(f"\n{Fore.YELLOW}{companion}:{Style.RESET_ALL}")
                for i, change in enumerate(changes, 1):
                    node_id = change.get('node_id', '')
                    value = change.get('value', 0)
                    speaker = change.get('speaker', '')
                    text = change.get('text', '')
                    
                    # Format the value with color
                    if value > 0:
                        value_str = f"{Fore.GREEN}+{value}{Style.RESET_ALL}"
                    elif value < 0:
                        value_str = f"{Fore.RED}{value}{Style.RESET_ALL}"
                    else:
                        value_str = f"{Fore.WHITE}{value}{Style.RESET_ALL}"
                    
                    # Truncate long dialog text
                    if len(text) > 70:
                        text = text[:67] + "..."
                    
                    # Print the change with context
                    print(f"  {i}. Node {node_id}: {value_str}")
                    print(f"     {speaker}: \"{text}\"")
        
        if not any_changes:
            print(f"{Fore.YELLOW}No approval changes recorded in this session.{Style.RESET_ALL}")

    def reset_state(self):
        """Reset the simulator state"""
        for companion in self.companion_approvals:
            self.companion_approvals[companion] = 0
            self.companion_approval_history[companion] = []
        self.visited_nodes = []
        self.active_flags = set(self.default_flags)
        #print(f"\n{Fore.GREEN}Simulator state reset.{Style.RESET_ALL}")

    def export_paths_to_txt(self, all_paths, output_file='dialog_paths.txt'):
        """Export all simulated dialog paths to a text file, with the custom format:
        "{speaker}: {text} || [context] {context} ||[approval] {list of approval changes, if exists}"
        For tagcinematic nodes: "[description] {text}".
        Each utterance on a separate line."""
        print(f"{Fore.GREEN}Exporting {len(all_paths)} dialog paths to {output_file}...{Style.RESET_ALL}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Baldur's Gate 3 Dialog Paths\n")
            f.write(f"Total paths: {len(all_paths)}\n\n")
            f.write(f"Synopsis: {self.metadata.get('synopsis', '')}\n")
            f.write(f"How to trigger: {self.metadata.get('how_to_trigger', '')}\n\n")
            
            for i, path in enumerate(all_paths, 1):
                f.write(f"Path {i}:\n")
                
                # Add custom formatted output for each node in the path
                for node_id in path:
                    if node_id in ["MAX_DEPTH_REACHED", "NODE_NOT_FOUND"]:
                        f.write(f"[{node_id}]\n")
                        continue
                        
                    node = self._get_node(node_id)
                    if not node:
                        continue
                        
                    speaker = node.get('speaker', 'Unknown')
                    text = self._get_effective_text(node)
                    context = node.get('context', '')
                    approvals = node.get('approval', [])
                    
                    # Skip nodes without text
                    if not text:
                        continue
                    
                    # Format the line according to requirements
                    line = ""
                    
                    # Handle speaker/text part based on node type
                    if node.get('node_type') == 'tagcinematic':
                        line = f"[description] {text}"
                    else:
                        line = f"{speaker}: {text}"
                    
                    # Add context if present
                    if context:
                        line += f" || [context] {context}"
                    
                    # Add approval changes if present
                    if approvals:
                        line += f" || [approval] {', '.join(approvals)}"
                    
                    # Write the formatted line
                    f.write(f"{line}\n")
                
                f.write("\n")  # Add extra line between paths
                
        print(f"{Fore.GREEN}Paths exported successfully to {output_file}{Style.RESET_ALL}")
        return output_file
        
    def create_traversal_data(self, all_paths):
        """Create a structured data representation of all traversals
        
        Returns:
            list: A list of traversals, each traversal is a list containing dictionaries with node data
        """
        print(f"{Fore.GREEN}Creating structured traversal data for {len(all_paths)} paths...{Style.RESET_ALL}")
        
        traversals = []
        
        for path in all_paths:
            traversal = []
            for node_id in path:
                if node_id in ["MAX_DEPTH_REACHED", "NODE_NOT_FOUND", "TIMEOUT_REACHED"]:
                    # Add special marker nodes
                    traversal.append({
                        "id": node_id,
                        "special_marker": True
                    })
                    continue
                    
                node = self._get_node(node_id)
                if not node:
                    # Handle case where node isn't found (e.g., broken link in path)
                    traversal.append({
                        "id": node_id,
                        "error": "NODE_DATA_NOT_FOUND",
                        "special_marker": True # Mark as special for easier filtering downstream
                    })
                    continue

                # Check if it's an alias node
                if node.get("node_type") == "alias":
                    target_id = node.get('link') # Assuming 'link' holds the target ID for aliases
                    target_node = self._get_node(target_id) if target_id else None

                    if target_node:
                        # Start with target node data, copying relevant fields
                        target_text = self._get_effective_text(target_node)
                        node_data = {
                            "id": node_id, # Use the original alias node ID for the path
                            "speaker": target_node.get('speaker', 'Unknown'),
                            "text": target_text,
                            "node_type": target_node.get('node_type', 'normal'), # Use target's type initially
                            "checkflags": target_node.get('checkflags', []),
                            "setflags": target_node.get('setflags', []),
                            "goto": target_node.get('goto', ''),
                            "link": target_node.get('link', ''), # Target's link, might be overridden
                            "is_end": target_node.get('is_end', False),
                            "approval": target_node.get('approval', []),
                            "context": target_node.get('context', '')
                        }
                        # Add a marker indicating resolution
                        node_data['resolved_from_alias'] = target_id

                        # Override with non-empty values from the alias node itself
                        override_fields = ['speaker', 'text', 'checkflags', 'setflags', 'goto', 'link', 'is_end', 'approval'] # Add 'rolls' etc. if needed
                        for field in override_fields:
                            alias_value = node.get(field)
                            is_non_empty = False
                            if isinstance(alias_value, (str, list)):
                                if alias_value: # Checks for non-empty string or list
                                    is_non_empty = True
                            elif field == 'is_end' and isinstance(alias_value, bool): # Override boolean if explicitly present in alias
                                is_non_empty = True # The presence of the boolean key itself is information
                            # Add checks for other types if necessary (e.g., int 0 might be a valid override)

                            if is_non_empty:
                                node_data[field] = alias_value
                                # If alias overrides the type-defining fields, reflect that? For now, keeps target type.

                        traversal.append(node_data)

                    else:
                        # Target node not found or no target_id, append raw alias info with an error
                        alias_text = self._get_effective_text(node)
                        node_data = {
                            "id": node_id,
                            "speaker": node.get('speaker', 'Unknown'),
                            "text": alias_text,
                            "node_type": node.get('node_type', 'alias'), # Keep type as alias
                            "checkflags": node.get('checkflags', []),
                            "setflags": node.get('setflags', []),
                            "goto": node.get('goto', ''),
                            "link": node.get('link', ''), # This is the target_id
                            "is_end": node.get('is_end', False),
                            "approval": node.get('approval', []),
                            "context": node.get('context', ''),
                            "error": f"ALIAS_TARGET_NOT_FOUND ({target_id})" if target_id else "ALIAS_TARGET_MISSING"
                        }
                        traversal.append(node_data)

                else:
                    # Not an alias node, create data as before
                    node_text = self._get_effective_text(node)
                    node_data = {
                        "id": node_id,
                        "speaker": node.get('speaker', 'Unknown'),
                        "text": node_text,
                        "node_type": node.get('node_type', 'normal'),
                        "checkflags": node.get('checkflags', []),
                        "setflags": node.get('setflags', []),
                        "goto": node.get('goto', ''),
                        "link": node.get('link', ''),
                        "is_end": node.get('is_end', False),
                        "approval": node.get('approval', []),
                        "context": node.get('context', '')
                    }
                    traversal.append(node_data)
            
            traversals.append(traversal)
        
        return traversals
        
    def export_traversals_to_json(self, traversals, output_file='traversals/dialog_traversals.json'):
        """Export structured traversal data to a JSON file"""
        print(f"{Fore.GREEN}Exporting {len(traversals)} traversals to {output_file}...{Style.RESET_ALL}")
        
        # Ensure output directory exists if provided
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(traversals, f, indent=2, ensure_ascii=False)
            
        print(f"{Fore.GREEN}Traversals exported successfully to {output_file}{Style.RESET_ALL}")
        return output_file
    def export_approval_history(self, output_file='approval_history.json'):
        """Export the approval history to a JSON file"""
        print(f"{Fore.GREEN}Exporting approval history to {output_file}...{Style.RESET_ALL}")
        
        # Create a structured version of the approval history
        history_data = {
            "companions": {},
            "summary": {}
        }
        
        for companion, changes in self.companion_approval_history.items():
            if changes:
                history_data["companions"][companion] = changes
                history_data["summary"][companion] = self.companion_approvals[companion]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
            
        print(f"{Fore.GREEN}Approval history exported successfully to {output_file}{Style.RESET_ALL}")
        return output_file

    def export_paths_to_dict(self, all_paths, output_file='dialog_dict.py'):
        """Export all simulated dialog paths to a Python dictionary
        
        This creates a Python file containing a dictionary where:
        - Keys are "path_1", "path_2", etc.
        - Values are strings with the full dialog text for each path
        
        Args:
            all_paths (list): List of node paths to export
            output_file (str): Output Python file path
            
        Returns:
            str: Path to the output file
        """
        print(f"{Fore.GREEN}Exporting {len(all_paths)} dialog paths to Python dictionary in {output_file}...{Style.RESET_ALL}")
        
        # Ensure output directory exists if provided
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Create the dictionary structure
        dialog_dict = {}
        
        for i, path in enumerate(all_paths, 1):
            path_key = f"path_{i}"
            dialog_text = []
            
            # Process each node in the path with custom formatting
            for node_id in path:
                if node_id in ["MAX_DEPTH_REACHED", "NODE_NOT_FOUND"]:
                    dialog_text.append(f"[{node_id}]")
                    continue
                    
                node = self._get_node(node_id)
                if not node:
                    continue
                    
                speaker = node.get('speaker', 'Unknown')
                text = self._get_effective_text(node)
                context = node.get('context', '')
                approvals = node.get('approval', [])
                
                # Skip nodes without text
                if not text:
                    continue
                
                # Format the line according to requirements
                line = ""
                
                # Handle speaker/text part based on node type
                if node.get('node_type') == 'tagcinematic':
                    line = f"[description] {text}"
                else:
                    line = f"{speaker}: {text}"
                
                # Add context if present
                if context:
                    line += f" || [context] {context}"
                
                # Add approval changes if present
                if approvals:
                    line += f" || [approval] {', '.join(approvals)}"
                
                dialog_text.append(line)
            
            # Join all lines with newlines to create a single string for this path
            dialog_dict[path_key] = "\n".join(dialog_text)
        
        # Write the dictionary to a Python file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Generated dialog paths dictionary\n\n")
            f.write("dialog_paths = {\n")
            
            for key, value in dialog_dict.items():
                # Format the multiline string properly for Python
                formatted_value = value.replace("'", "\\'")  # Escape single quotes
                f.write(f"    '{key}': '''\n{formatted_value}\n''',\n\n")
            
            f.write("}\n")
        
        print(f"{Fore.GREEN}Dialog paths dictionary exported successfully to {output_file}{Style.RESET_ALL}")
        return output_file

    def execute_path(self, path, initial_flags=None):
        """
        Executes a specific dialog path, applying setflags and approvals.
        Does NOT check flags during traversal, assumes path validity.
        Resets and uses provided initial flags.

        Args:
            path (list): List of node IDs representing the path to execute.
            initial_flags (set, optional): Flags to start with. If None, uses defaults.

        Returns:
            tuple: (list_of_node_data, final_flags_set)
                   - list_of_node_data: Detailed data for each node visited.
                   - final_flags_set: The set of active flags after traversing the path.
        """
        if not path:
            return [], initial_flags if initial_flags else set(self.default_flags)

        # Reset state but keep initial flags
        self.reset_state() # Resets approvals and visited nodes
        if initial_flags is not None:
             self.set_initial_flags(initial_flags)
        else:
             self.active_flags = set(self.default_flags) # Ensure defaults if none provided

        # print(f"{Fore.CYAN}Executing path: {path} with initial flags: {self.active_flags}{Style.RESET_ALL}")

        traversed_nodes_data = []
        current_active_flags = self.active_flags.copy() # Work with a copy locally

        for node_id in path:
            if node_id in ["MAX_DEPTH_REACHED", "NODE_NOT_FOUND"]:
                traversed_nodes_data.append({
                    "id": node_id,
                    "special_marker": True
                })
                continue

            # Get node data directly, do not follow jumps/links here as path is pre-determined
            # However, the path provided SHOULD already have jumps/links resolved if needed.
            node_data = self._get_node(node_id)

            if not node_data:
                print(f"{Fore.RED}Node {node_id} not found during path execution. Skipping.{Style.RESET_ALL}")
                traversed_nodes_data.append({
                    "id": node_id,
                    "error": "NODE_DATA_NOT_FOUND",
                    "special_marker": True
                })
                continue

            # Process approvals (affects internal simulator state)
            self._process_approvals(node_data) # Uses self.companion_approvals etc.

            # Process setflags (affects the flags being tracked for return)
            # Logic copied and adapted from self._process_setflags
            for flag in node_data.get('setflags', []):
                if "= False" in flag:
                    flag_to_remove = flag.split('= False')[0].strip()
                    if flag_to_remove in current_active_flags:
                        current_active_flags.remove(flag_to_remove)
                else:
                    current_active_flags.add(flag.strip())

            # Store node data for the result (without mutating the node)
            effective_text = self._get_effective_text(node_data)
            traversed_nodes_data.append({
                "id": node_id,
                "speaker": node_data.get('speaker', 'Unknown'),
                "text": effective_text,
                "node_type": node_data.get('node_type', 'normal'),
                "checkflags": node_data.get('checkflags', []), # Include for info
                "setflags": node_data.get('setflags', []),   # Include for info
                "goto": node_data.get('goto', ''),
                "link": node_data.get('link', ''),
                "is_end": node_data.get('is_end', False),
                "approval": node_data.get('approval', []),
                "context": node_data.get('context', '')
            })

            # Update the main simulator flags (might be useful if called interactively, but primary return is separate)
            self.active_flags = current_active_flags.copy()

        # print(f"{Fore.GREEN}Path execution finished. Final flags: {current_active_flags}{Style.RESET_ALL}")
        return traversed_nodes_data, current_active_flags

    def execute_path_with_snapshots(self, path, initial_flags=None):
        """
        Executes a specific dialog path like execute_path, but also returns
        the snapshot of active flags after each node.

        Args:
            path (list): Node IDs to execute in order
            initial_flags (set|list|tuple|None): Starting flags

        Returns:
            tuple: (nodes_data, flags_per_step)
                   nodes_data: list of node data dicts per visited node
                   flags_per_step: list of sets, active flags AFTER each node
        """
        nodes_data, _ = self.execute_path(path, initial_flags)

        # Re-apply to collect snapshots per step deterministically
        # because execute_path returns only final flags
        if initial_flags is not None:
            if isinstance(initial_flags, set):
                current_flags = initial_flags.copy()
            elif isinstance(initial_flags, (list, tuple)):
                current_flags = set(initial_flags)
            else:
                current_flags = set(self.default_flags)
        else:
            current_flags = set(self.default_flags)

        flags_per_step = []

        for node_data in nodes_data:
            if node_data.get('special_marker'):
                flags_per_step.append(current_flags.copy())
                continue

            # Apply setflags like in execute_path
            for flag in node_data.get('setflags', []):
                if "= False" in flag:
                    flag_to_remove = flag.split('= False')[0].strip()
                    if flag_to_remove in current_flags:
                        current_flags.remove(flag_to_remove)
                else:
                    current_flags.add(flag.strip())

            flags_per_step.append(current_flags.copy())

        return nodes_data, flags_per_step

    def _add_nodes_to_graph(self, dot, node_id, visited, current_depth, max_depth):
        """Recursively add nodes and edges to the Graphviz graph."""
        if node_id in visited or current_depth > max_depth:
            return

        node = self._get_node(node_id)
        if not node:
            # Add a placeholder for missing nodes
            if node_id not in visited:
                 dot.node(node_id, label=f"{node_id}\n(Not Found)", shape='box', style='filled', fillcolor='red')
                 visited.add(node_id)
            return

        visited.add(node_id)

        # Node styling
        speaker = node.get('speaker', 'Unknown')
        text_preview = node.get('text', node.get('context', ''))[:40] # Limit text length
        if len(node.get('text', node.get('context', ''))) > 40:
            text_preview += "..."
        label = f"{node_id}\n{speaker}\n'{text_preview}'" # Use \n for newline in graphviz label
        shape = 'box'
        style = 'filled'
        fillcolor = 'lightgrey'
        node_color = 'black' # Border color

        if speaker == 'Player':
            fillcolor = 'lightblue'
        elif node.get('node_type') == 'jump':
            shape = 'cds'
            fillcolor = 'yellow'
        elif node.get('node_type') == 'tagcinematic':
            shape = 'note'
            fillcolor = 'lightgoldenrod'
        elif node.get('node_type') == 'alias':
            shape = 'hexagon'
            fillcolor = 'lightcoral'


        if node.get('is_end', False):
            node_color = 'red' # Red border for end nodes
            style += ',bold'

        dot.node(node_id, label=label, shape=shape, style=style, fillcolor=fillcolor, color=node_color)

        # Process children
        children = node.get('children', {})
        for child_id in children:
            # Add edge first
            dot.edge(node_id, child_id, label='child', color='black')
            # Recurse only if child node exists (avoid infinite loop on bad data)
            if self._get_node(child_id):
                 self._add_nodes_to_graph(dot, child_id, visited, current_depth + 1, max_depth)
            elif child_id not in visited: # Add error node if child doesn't exist and hasn't been added
                 dot.node(child_id, label=f"{child_id}\n(Child Not Found)", shape='box', style='filled', fillcolor='red')
                 visited.add(child_id)


        # Process goto
        goto_id = node.get('goto')
        if goto_id:
            # Add edge first
            dot.edge(node_id, goto_id, label='goto', style='dashed', color='blue')
            # Recurse only if goto node exists
            if self._get_node(goto_id):
                self._add_nodes_to_graph(dot, goto_id, visited, current_depth + 1, max_depth)
            elif goto_id not in visited: # Add error node if goto doesn't exist
                dot.node(goto_id, label=f"{goto_id}\n(Goto Target Not Found)", shape='box', style='filled', fillcolor='red')
                visited.add(goto_id)


        # Process link
        link_id = node.get('link')
        if link_id:
            # Add edge first
            dot.edge(node_id, link_id, label='link', style='dotted', color='green')
            # Recurse only if link node exists
            if self._get_node(link_id):
                self._add_nodes_to_graph(dot, link_id, visited, current_depth + 1, max_depth)
            elif link_id not in visited: # Add error node if link doesn't exist
                dot.node(link_id, label=f"{link_id}\n(Link Target Not Found)", shape='box', style='filled', fillcolor='red')
                visited.add(link_id)

    def visualize_structure(self, output_filename='dialog_structure', start_node_id=None, max_depth=10, render_format='pdf'):
        """
        Generates a visualization of the dialog tree structure using Graphviz.

        Args:
            output_filename (str): The base name for the output file (without extension).
            start_node_id (str, optional): The node ID to start visualization from. If None, visualizes from all root nodes. Defaults to None.
            max_depth (int): Maximum depth to visualize. Defaults to 10.
            render_format (str): The output format (e.g., 'pdf', 'png', 'svg'). Defaults to 'pdf'.

        Returns:
            str: The path to the generated visualization file, or None if generation failed.
        """
        if not GRAPHVIZ_AVAILABLE:
            print(f"{Fore.RED}Graphviz is not available. Cannot generate visualization.{Style.RESET_ALL}")
            return None

        print(f"\n{Fore.CYAN}Generating dialog structure visualization (max depth: {max_depth}, format: {render_format})...{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Output base filename: {output_filename}{Style.RESET_ALL}")

        # Extract directory from filename if present
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
             try:
                 os.makedirs(output_dir)
                 print(f"{Fore.GREEN}Created output directory: {output_dir}{Style.RESET_ALL}")
             except OSError as e:
                 print(f"{Fore.RED}Error creating directory {output_dir}: {e}{Style.RESET_ALL}")
                 return None
        elif not output_dir:
            output_dir = '.' # Ensure filename has a directory part for graphviz

        # Graphviz uses filename for the .gv file, needs directory
        gv_filepath_base = os.path.join(output_dir, os.path.basename(output_filename))


        dot = graphviz.Digraph(
            comment=f'Dialog Structure - {self.metadata.get("synopsis", "Unknown Dialog")}',
            graph_attr={'rankdir': 'TB', 'splines': 'ortho'}, # Try ortho splines
            node_attr={'fontsize': '10'},
            edge_attr={'fontsize': '8'}
        )

        visited = set()

        try:
            if start_node_id:
                if self._get_node(start_node_id):
                    print(f"Starting visualization from node: {start_node_id}")
                    self._add_nodes_to_graph(dot, start_node_id, visited, 0, max_depth)
                else:
                    print(f"{Fore.RED}Error: Start node ID '{start_node_id}' not found.{Style.RESET_ALL}")
                    return None
            else:
                print(f"Starting visualization from {len(self.root_nodes)} root nodes.")
                for root_id in self.root_nodes:
                    self._add_nodes_to_graph(dot, root_id, visited, 0, max_depth)

            # Render the graph
            # 'cleanup=True' removes the intermediate .gv file
            rendered_path = dot.render(gv_filepath_base, format=render_format, view=False, cleanup=True)

            print(f"{Fore.GREEN}Visualization successfully generated: {rendered_path}{Style.RESET_ALL}")
            return rendered_path

        except graphviz.backend.execute.ExecutableNotFound:
            print(f"{Fore.RED}Error: Graphviz executable not found.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Please ensure Graphviz is installed and in your system's PATH.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Download: https://graphviz.org/download/{Style.RESET_ALL}")
            # Optionally save the .gv source file for manual rendering
            gv_source_path = gv_filepath_base + '.gv'
            try:
                dot.save(gv_source_path)
                print(f"{Fore.YELLOW}Graphviz source file saved: {gv_source_path}. You can render it manually.{Style.RESET_ALL}")
            except Exception as e_save:
                 print(f"{Fore.RED}Failed to save Graphviz source file: {e_save}{Style.RESET_ALL}")

            return None
        except Exception as e:
            print(f"{Fore.RED}An error occurred during visualization generation:{Style.RESET_ALL}")
            print(traceback.format_exc())
             # Optionally save the .gv source file for debugging
            gv_source_path = gv_filepath_base + '.gv'
            try:
                dot.save(gv_source_path)
                print(f"{Fore.YELLOW}Graphviz source file saved for debugging: {gv_source_path}{Style.RESET_ALL}")
            except Exception as e_save:
                 print(f"{Fore.RED}Failed to save Graphviz source file: {e_save}{Style.RESET_ALL}")
            return None

    def build_qa_examples_from_path(self, path, initial_flags=None, relevant_flag_prefixes=None):
        """
        Build QA examples from a single node path, where each example corresponds
        to an approval node and includes the prior dialogue (within this path)
        as context plus a compact snapshot of relevant flags.

        - relevant_flag_prefixes: optional list of string prefixes used to filter
          the active flags to those that cluster context (e.g., quest, companion,
          location flags). If None, uses all active flags.

        Returns: list of QA example dicts
        Each example:
          {
            'approval_node_id': str,
            'context_dialogue': [ {speaker,text,context,node_id} ... up to this node],
            'approval_list': [...],
            'active_flags': [...],          # filtered and sorted
            'full_active_flags_count': int  # diagnostic count
          }
        """
        nodes_data, flags_per_step = self.execute_path_with_snapshots(path, initial_flags)

        qa_examples = []
        for idx, node_data in enumerate(nodes_data):
            if node_data.get('special_marker'):
                continue
            approvals = node_data.get('approval', [])
            if not approvals:
                continue

            # Build context dialogue window within this single dialogue path
            context_dialogue = []
            for prior in nodes_data[: idx + 1]:
                if prior.get('special_marker'):
                    continue
                context_dialogue.append({
                    'node_id': prior['id'],
                    'speaker': prior.get('speaker', 'Unknown'),
                    'text': prior.get('text', ''),
                    'context': prior.get('context', ''),
                })

            # Determine relevant flags by filtering snapshot
            snapshot_flags = flags_per_step[idx] if idx < len(flags_per_step) else set()
            if relevant_flag_prefixes:
                filtered = set()
                for flag in snapshot_flags:
                    for pref in relevant_flag_prefixes:
                        if flag.startswith(pref):
                            filtered.add(flag)
                            break
                active_flags_list = sorted(filtered)
            else:
                active_flags_list = sorted(snapshot_flags)

            qa_examples.append({
                'approval_node_id': node_data['id'],
                'context_dialogue': context_dialogue,
                'approval_list': approvals,
                'active_flags': active_flags_list,
                'full_active_flags_count': len(snapshot_flags),
            })

        return qa_examples

    def export_qa_dataset(self, qa_examples, output_file='qa/approval_qa.json'):
        """Export QA examples to JSON, ensuring directory exists."""
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(qa_examples, f, indent=2, ensure_ascii=False)
        print(f"{Fore.GREEN}QA dataset exported successfully to {output_file}{Style.RESET_ALL}")
        return output_file


    # ========================= RAG / FAISS FEATURES =========================
    def _ensure_embedder(self, model_name=None):
        """Ensure an embedder backend is available.

        Returns a tuple (backend, model_name) where backend is 'openai' or 'sbert'.
        """
        # Prefer OpenAI via LiteLLM
        if LITELLM_EMBED_AVAILABLE:
            name = model_name or DEFAULT_EMBED_MODEL
            return ('openai', name)
        # Fallback to sentence-transformers
        if not SENT_EMBED_AVAILABLE:
            return (None, None)
        if self._embedder is None:
            name = model_name or DEFAULT_EMBED_MODEL
            try:
                self._embedder = SentenceTransformer(name)
            except Exception as e:
                print(f"{Fore.RED}Failed to load embedding model '{name}': {e}{Style.RESET_ALL}")
                self._embedder = None
                return (None, None)
        return ('sbert', model_name or DEFAULT_EMBED_MODEL)

    def _ensure_faiss(self, dim):
        """Ensure a FAISS index exists with the given dimensionality."""
        if not FAISS_AVAILABLE:
            return None
        if self._faiss_index is None:
            try:
                self._faiss_index = faiss.IndexFlatIP(dim)
                self._faiss_dim = dim
                self._faiss_id_to_meta = {}
            except Exception as e:
                print(f"{Fore.RED}Failed to initialize FAISS index: {e}{Style.RESET_ALL}")
                self._faiss_index = None
        elif self._faiss_dim != dim:
            print(f"{Fore.YELLOW}Reinitializing FAISS index to dimension {dim}{Style.RESET_ALL}")
            try:
                self._faiss_index = faiss.IndexFlatIP(dim)
                self._faiss_dim = dim
                self._faiss_id_to_meta = {}
            except Exception as e:
                print(f"{Fore.RED}Failed to reinitialize FAISS index: {e}{Style.RESET_ALL}")
                self._faiss_index = None
        return self._faiss_index

    def _embed_texts(self, texts, model_name=None):
        """Return L2-normalized embeddings for a list of texts (for cosine via inner product)."""
        backend, model_used = self._ensure_embedder(model_name)
        if backend is None:
            return None
        try:
            if backend == 'openai':
                # LiteLLM: batch embed
                embeddings = []
                # Some providers require batching; do small batches
                batch_size = 64
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    resp = litellm_embedding(model=model_used, input=batch)
                    # OpenAI-like response structure
                    batch_vecs = [d['embedding'] for d in resp['data']]
                    embeddings.extend(batch_vecs)
                arr = np.array(embeddings, dtype='float32')
                # Normalize for cosine similarity via inner product
                norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
                arr = arr / norms
                return arr
            else:
                vecs = self._embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
                return vecs.astype('float32')
        except Exception as e:
            print(f"{Fore.RED}Embedding failed: {e}{Style.RESET_ALL}")
            return None

    def save_cluster_contexts_to_faiss(self, cluster_index_file='goals_to_json_paths.json', contexts_root='qa-contexts', include_synopsis_fallback=True, model_name=None):
        """Save synopsis or generated contexts for the current file's cluster into FAISS.

        Adds/overwrites an in-memory FAISS index with one vector per dialogue file in the same cluster
        (filtered by Act ordering not later than current). Uses generated context if available under
        contexts_root; otherwise optionally falls back to the raw synopsis.
        """
        if not FAISS_AVAILABLE or not SENT_EMBED_AVAILABLE:
            print(f"{Fore.YELLOW}FAISS or embeddings unavailable; cannot save to vector DB.{Style.RESET_ALL}")
            return False

        entries = []
        # Load all clusters to know which clusters each path belongs to
        try:
            with open(cluster_index_file, 'r', encoding='utf-8') as f:
                clusters = json.load(f)
        except Exception as e:
            print(f"{Fore.RED}Failed to read cluster index: {e}{Style.RESET_ALL}")
            return False

        # Collect items (contexts or synopses), along with their cluster memberships
        def _collect_items(use_ctx):
            items = []
            for cluster_name, paths in clusters.items():
                for p in paths:
                    # Filter by Act (not later than current)
                    # Reuse act filter via _gather_cluster_synopses helpers
                    # We'll assemble text directly to keep cluster membership
                    # Try to read context if requested
                    text = None
                    if use_ctx:
                        # Map to qa-contexts
                        norm = os.path.normpath(p)
                        output_prefix = 'output' + os.sep
                        rel = None
                        if norm.startswith(output_prefix):
                            rel = norm[len(output_prefix):]
                        else:
                            marker = os.sep + 'output' + os.sep
                            idx = norm.find(marker)
                            if idx != -1:
                                rel = norm[idx + len(marker):]
                        if rel is None:
                            rel = os.path.basename(norm)
                        rel_dir = os.path.dirname(rel)
                        stem = os.path.splitext(os.path.basename(rel))[0]
                        ctx_path = os.path.join(contexts_root, rel_dir, f"{stem}_context.txt")
                        try:
                            with open(ctx_path, 'r', encoding='utf-8') as fctx:
                                text = fctx.read().strip()
                        except Exception:
                            text = None

                    if not text and include_synopsis_fallback:
                        try:
                            with open(p, 'r', encoding='utf-8') as fj:
                                data = json.load(fj)
                                text = data.get('metadata', {}).get('synopsis', '')
                        except Exception:
                            text = None

                    if text and text.strip():
                        items.append((p, text.strip(), cluster_name))
            return items

        items = _collect_items(use_ctx=True)
        if not items and include_synopsis_fallback:
            items = _collect_items(use_ctx=False)

        for path_str, text in items:
            if text and text.strip():
                entries.append((path_str, text.strip()))

        if not entries:
            print(f"{Fore.YELLOW}No cluster contexts/synopses found to index.{Style.RESET_ALL}")
            return False

        texts = [t for _, t in entries]
        vecs = self._embed_texts(texts, model_name=model_name)
        if vecs is None:
            return False

        index = self._ensure_faiss(vecs.shape[1])
        if index is None:
            return False

        # Reset index and add vectors
        try:
            if index.ntotal > 0:
                # Recreate to clear
                self._faiss_index = faiss.IndexFlatIP(vecs.shape[1])
                index = self._faiss_index
                self._faiss_id_to_meta = {}
                self._faiss_path_to_id = {}
                self._faiss_dim = vecs.shape[1]
            index.add(vecs)
            # Map ids
            # Build mapping entries: if a path appears in multiple clusters, keep one vector but list both clusters
            # Merge clusters per path
            path_to_clusters = {}
            for pp, _, c in items:
                path_to_clusters.setdefault(os.path.abspath(pp), set()).add(c)

            for i, (p, _) in enumerate(entries):
                abs_p = os.path.abspath(p)
                self._faiss_id_to_meta[i] = {
                    'path': p,
                    'clusters': sorted(list(path_to_clusters.get(abs_p, set())))
                }
                self._faiss_path_to_id[abs_p] = i
                self._faiss_path_to_ids.setdefault(abs_p, []).append(i)
            print(f"{Fore.GREEN}Indexed {len(entries)} cluster texts into FAISS (dim={self._faiss_dim}).{Style.RESET_ALL}")
            return True
        except Exception as e:
            print(f"{Fore.RED}Failed to add vectors to FAISS: {e}{Style.RESET_ALL}")
            return False

    def retrieve_relevant_cluster_contexts(self, query_text, top_k=5, model_name=None, restrict_to_current_clusters=False, cluster_index_file='goals_to_json_paths.json', prefer_source_context=True, contexts_root='qa-contexts'):
        """Retrieve top-k most similar cluster contexts from FAISS using cosine similarity.

        Similarity is computed between the query vector (built from the current source synopsis or
        generated context + brief dialogue) and the stored vectors for synopses/contexts.

        If restrict_to_current_clusters=True, limits candidates to entries whose cluster names
        include at least one cluster that contains self.json_path in 'goals_to_json_paths.json'.
        """
        if not FAISS_AVAILABLE or not SENT_EMBED_AVAILABLE or self._faiss_index is None:
            return []
        # Build query text either from synopsis/context or provided query_text
        if not query_text:
            query_text = self._get_current_source_text(prefer_context=prefer_source_context, contexts_root=contexts_root)
        qvec = self._embed_texts([query_text], model_name=model_name)
        if qvec is None:
            return []
        try:
            # If no restriction, do a normal top_k search
            if not restrict_to_current_clusters:
                D, I = self._faiss_index.search(qvec, top_k)
                results = []
                for idx, score in zip(I[0], D[0]):
                    if idx == -1:
                        continue
                    meta = self._faiss_id_to_meta.get(int(idx), {})
                    results.append({'path': meta.get('path', ''), 'score': float(score)})
                return results

            # Determine clusters of current path
            current_clusters = set()
            try:
                with open(cluster_index_file, 'r', encoding='utf-8') as f:
                    clusters = json.load(f)
                current_abs = os.path.abspath(self.json_path)
                for cname, paths in clusters.items():
                    for p in paths:
                        if os.path.abspath(p) == current_abs:
                            current_clusters.add(cname)
                            break
            except Exception:
                current_clusters = set()

            if not current_clusters:
                D, I = self._faiss_index.search(qvec, top_k)
                results = []
                for idx, score in zip(I[0], D[0]):
                    if idx == -1:
                        continue
                    meta = self._faiss_id_to_meta.get(int(idx), {})
                    results.append({'path': meta.get('path', ''), 'score': float(score)})
                return results

            # Retrieve a larger candidate pool, then filter to current clusters
            pool_k = min(max(top_k * 10, 50), max(self._faiss_index.ntotal, top_k))
            D, I = self._faiss_index.search(qvec, pool_k)
            candidates = []
            for idx, score in zip(I[0], D[0]):
                if idx == -1:
                    continue
                meta = self._faiss_id_to_meta.get(int(idx), {})
                meta_clusters = set(meta.get('clusters', []) or [])
                if meta_clusters & current_clusters:
                    candidates.append({'path': meta.get('path', ''), 'score': float(score)})
            # Take top_k from filtered
            candidates.sort(key=lambda x: x['score'], reverse=True)
            return candidates[:top_k]
        except Exception as e:
            print(f"{Fore.RED}FAISS search failed: {e}{Style.RESET_ALL}")
            return []

    def save_faiss_to_disk(self, dir_path='vector_db'):
        """Persist the FAISS index and its metadata to disk for inspection/debugging.

        Writes:
          - {dir_path}/index.faiss
          - {dir_path}/meta.json (id->path, clusters, dim)
        """
        if not FAISS_AVAILABLE:
            print(f"{Fore.YELLOW}FAISS not available; cannot persist index.{Style.RESET_ALL}")
            return False
        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            print(f"{Fore.YELLOW}No FAISS index in memory to save.{Style.RESET_ALL}")
            return False
        try:
            os.makedirs(dir_path, exist_ok=True)
            index_path = os.path.join(dir_path, 'index.faiss')
            meta_path = os.path.join(dir_path, 'meta.json')

            faiss.write_index(self._faiss_index, index_path)

            entries = []
            for id_str, meta in self._faiss_id_to_meta.items():
                try:
                    idx_int = int(id_str)
                except Exception:
                    idx_int = id_str
                entries.append({
                    'id': idx_int,
                    'path': meta.get('path', ''),
                    'clusters': meta.get('clusters', []),
                })
            payload = {
                'dim': self._faiss_dim,
                'entries': entries,
            }
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            print(f"{Fore.GREEN}Saved FAISS index to {index_path} and metadata to {meta_path}{Style.RESET_ALL}")
            return True
        except Exception as e:
            print(f"{Fore.RED}Failed to save FAISS index to disk: {e}{Style.RESET_ALL}")
            return False

    def load_faiss_from_disk(self, dir_path='vector_db'):
        """Load the FAISS index and metadata from disk into memory."""
        if not FAISS_AVAILABLE:
            print(f"{Fore.YELLOW}FAISS not available; cannot load index.{Style.RESET_ALL}")
            return False
        index_path = os.path.join(dir_path, 'index.faiss')
        meta_path = os.path.join(dir_path, 'meta.json')
        if not os.path.isfile(index_path) or not os.path.isfile(meta_path):
            print(f"{Fore.YELLOW}No persisted FAISS index found under {dir_path}.{Style.RESET_ALL}")
            return False
        try:
            index = faiss.read_index(index_path)
            with open(meta_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            self._faiss_index = index
            self._faiss_dim = int(payload.get('dim', 0)) or None
            self._faiss_id_to_meta = {}
            self._faiss_path_to_id = {}
            for entry in payload.get('entries', []):
                idx = int(entry.get('id', -1))
                path_val = entry.get('path', '')
                clusters = entry.get('clusters', [])
                self._faiss_id_to_meta[idx] = {'path': path_val, 'clusters': clusters}
                self._faiss_path_to_id[os.path.abspath(path_val)] = idx
            print(f"{Fore.GREEN}Loaded FAISS index from {index_path} with {self._faiss_index.ntotal} vectors.{Style.RESET_ALL}")
            return True
        except Exception as e:
            print(f"{Fore.RED}Failed to load FAISS index: {e}{Style.RESET_ALL}")
            return False

    def build_global_synopsis_faiss(self, cluster_index_file='goals_to_json_paths.json', model_name=None, save_dir='vector_db_synopses'):
        """Index ALL sessions' synopses from goals_to_json_paths.json into a FAISS DB.

        - Uses cosine similarity (normalized embeddings + inner product index)
        - Stores cluster memberships for each path in the metadata
        - Persists to disk under save_dir (index.faiss, meta.json)
        """
        if not FAISS_AVAILABLE or not SENT_EMBED_AVAILABLE:
            print(f"{Fore.YELLOW}FAISS or embeddings unavailable; cannot build global synopsis index.{Style.RESET_ALL}")
            return False

        # Read cluster index and build path -> clusters mapping
        try:
            with open(cluster_index_file, 'r', encoding='utf-8') as f:
                clusters = json.load(f)
        except Exception as e:
            print(f"{Fore.RED}Failed to read cluster index: {e}{Style.RESET_ALL}")
            return False

        path_to_clusters = {}
        for cname, paths in clusters.items():
            for p in paths:
                ap = os.path.abspath(p)
                path_to_clusters.setdefault(ap, set()).add(cname)

        # Collect synopses for all unique paths
        entries = []  # (abs_path, synopsis)
        for ap in sorted(path_to_clusters.keys()):
            try:
                with open(ap, 'r', encoding='utf-8') as fj:
                    data = json.load(fj)
                    syn = data.get('metadata', {}).get('synopsis', '')
                    if syn and syn.strip():
                        entries.append((ap, syn.strip()))
            except Exception:
                continue

        if not entries:
            print(f"{Fore.YELLOW}No synopses found to index from {cluster_index_file}.{Style.RESET_ALL}")
            return False

        texts = [t for _, t in entries]
        vecs = self._embed_texts(texts, model_name=model_name)
        if vecs is None:
            return False

        index = self._ensure_faiss(vecs.shape[1])
        if index is None:
            return False

        try:
            # Reset and add
            self._faiss_index = faiss.IndexFlatIP(vecs.shape[1])
            index = self._faiss_index
            index.add(vecs)

            # Build metadata maps
            self._faiss_id_to_meta = {}
            self._faiss_path_to_id = {}
            self._faiss_path_to_ids = {}
            self._faiss_dim = vecs.shape[1]
            for i, (ap, _) in enumerate(entries):
                clusters_list = sorted(list(path_to_clusters.get(ap, set())))
                self._faiss_id_to_meta[i] = {
                    'path': ap,
                    'clusters': clusters_list,
                }
                self._faiss_path_to_id[ap] = i
                self._faiss_path_to_ids.setdefault(ap, []).append(i)

            # Persist to disk for inspection
            self.save_faiss_to_disk(save_dir)
            print(f"{Fore.GREEN}Global synopsis FAISS built with {len(entries)} entries.{Style.RESET_ALL}")
            return True
        except Exception as e:
            print(f"{Fore.RED}Failed to build global synopsis FAISS: {e}{Style.RESET_ALL}")
            return False

    def build_query_from_current_dialogue(self, dialogue_samples):
        """Construct a concise query text from current dialogue samples and synopsis."""
        lines = []
        if dialogue_samples and isinstance(dialogue_samples, list):
            samples = [dialogue_samples] if dialogue_samples and isinstance(dialogue_samples[0], dict) else (dialogue_samples or [])
            for sample in samples[:1]:  # one sample keeps it concise
                for turn in sample[:6]:  # first few turns
                    txt = turn.get('text', '') or turn.get('context', '')
                    if txt:
                        lines.append(txt)
        syn = self.metadata.get('synopsis', '')
        if syn:
            lines.append(syn)
        return " \n".join(lines)[:2000]

    def _get_current_source_text(self, prefer_context=True, contexts_root='qa-contexts'):
        """Return the current file's generated context (preferred) or synopsis as text."""
        src_text = ''
        if prefer_context:
            try:
                norm = os.path.normpath(self.json_path)
                output_prefix = 'output' + os.sep
                rel = None
                if norm.startswith(output_prefix):
                    rel = norm[len(output_prefix):]
                else:
                    marker = os.sep + 'output' + os.sep
                    idx = norm.find(marker)
                    if idx != -1:
                        rel = norm[idx + len(marker):]
                if rel is None:
                    rel = os.path.basename(norm)
                rel_dir = os.path.dirname(rel)
                stem = os.path.splitext(os.path.basename(rel))[0]
                ctx_path = os.path.join(contexts_root, rel_dir, f"{stem}_context.txt")
                with open(ctx_path, 'r', encoding='utf-8') as f:
                    tmp = f.read().strip()
                    if tmp:
                        src_text = tmp
            except Exception:
                src_text = ''
        if not src_text:
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    src_text = data.get('metadata', {}).get('synopsis', '') or ''
            except Exception:
                src_text = ''
        return src_text

    def generate_context_with_rag(self, synopsis, dialogue_samples, model='gpt-5-mini', temperature=0.2, max_tokens=8000, top_k=5, cluster_index_file='goals_to_json_paths.json', contexts_root='qa-contexts', model_name=None, restrict_to_current_clusters=True, vector_db_dir='vector_db_synopses'):
        """RAG: Load an already-saved FAISS index, retrieve top-k related contexts, then call LLM.

        This method does NOT build or modify the index; it only loads from 'vector_db_dir'.
        """
        ok = True
        if self._faiss_index is None:
            ok = self.load_faiss_from_disk(vector_db_dir)
        retrieved = []
        if ok:
            query = self.build_query_from_current_dialogue(dialogue_samples)
            hits = self.retrieve_relevant_cluster_contexts(
                query,
                top_k=top_k,
                model_name=model_name,
                restrict_to_current_clusters=restrict_to_current_clusters,
                cluster_index_file=cluster_index_file,
                prefer_source_context=False,  # default to synopsis-based retrieval
                contexts_root=contexts_root
            )
            # Load text for hits
            self.last_retrieved_sessions = []
            self.last_retrieved_synopses = []
            for h in hits:
                p = h['path']
                # Try context file first
                ctx_items = self._gather_cluster_synopses(cluster_index_file=cluster_index_file, include_current=True, include_prior=True, include_future=False, limit=1000, use_context_outputs=True, contexts_root=contexts_root)
                text = None
                for cp, ctext in ctx_items:
                    if os.path.abspath(cp) == os.path.abspath(p):
                        text = ctext
                        break
                if text is None:
                    # Fallback to synopsis
                    try:
                        with open(p, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            text = data.get('metadata', {}).get('synopsis', '')
                    except Exception:
                        text = ''
                if text:
                    retrieved.append((p, text))
                    self.last_retrieved_sessions.append(p)
                    self.last_retrieved_synopses.append(text)

        return self.generate_context_with_llm(
            synopsis=synopsis,
            dialogue_samples=dialogue_samples,
            active_flags=None,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            cluster_synopses=retrieved if retrieved else None
        )

def build_qa_dataset_from_clusters(cluster_index_file='goals_to_json_paths.json',
                                   relevant_flag_prefixes=None,
                                   export_file='qa/cluster_approval_qa.json',
                                   test_mode=True,
                                   max_depth=20,
                                   verbose=False):
    """Build a QA dataset using a clustering of related dialogues.

    The cluster index JSON should map a cluster key to an ordered list of dialogue JSON paths.
    For each cluster, we process dialogues in order, maintaining an evolving set of
    cluster-level flags derived from prior dialogues. Each approval QA example in a
    later dialogue is built with initial flags seeded from prior dialogues in the cluster.

    Args:
        cluster_index_file (str): Path to JSON file of {cluster_key: [json_paths,...]}
        relevant_flag_prefixes (list[str]|None): Prefixes for filtering contextual flags
        export_file (str): Output JSON for aggregated QA examples
        test_mode (bool): Bypass flag requirements during path discovery
        max_depth (int): Traversal depth for simulations
        verbose (bool): Print progress

    Returns:
        str: Path to exported QA dataset JSON
    """
    if not os.path.isfile(cluster_index_file):
        print(f"{Fore.RED}Cluster index file not found: {cluster_index_file}{Style.RESET_ALL}")
        return None

    with open(cluster_index_file, 'r', encoding='utf-8') as f:
        clusters = json.load(f)

    all_qa = []

    def _collect_flag_effects_from_paths(simulator, paths):
        """Collect union of flags set True and flags set False across provided paths."""
        flags_true = set()
        flags_false = set()
        for path in paths:
            nodes_data, _ = simulator.execute_path(path, initial_flags=simulator.default_flags)
            for node in nodes_data:
                for flag in node.get('setflags', []):
                    if "= False" in flag:
                        flags_false.add(flag.split('= False')[0].strip())
                    else:
                        flags_true.add(flag.strip())
        return flags_true, flags_false

    for cluster_key, dialog_paths in clusters.items():
        if verbose:
            print(f"{Fore.CYAN}Processing cluster: {cluster_key} ({len(dialog_paths)} files){Style.RESET_ALL}")

        # Cluster-level evolving flags (start with simulator defaults)
        cluster_flags = set()

        for json_path in dialog_paths:
            if verbose:
                print(f"{Fore.BLUE}  Dialogue file: {json_path}{Style.RESET_ALL}")
            if not os.path.isfile(json_path):
                print(f"{Fore.YELLOW}  Skipping missing file: {json_path}{Style.RESET_ALL}")
                continue

            sim = DialogSimulator(json_path)

            # Discover approval paths
            approval_paths, _, _, _ = sim.simulate_approval_paths(
                max_depth=max_depth,
                print_paths=False,
                test_mode=test_mode,
                export_txt=False,
                export_json=False,
                export_dict=False,
                verbose=verbose,
            )

            # Build QA per approval path using current cluster flags as initial context
            for path in approval_paths:
                qa_examples = sim.build_qa_examples_from_path(
                    path,
                    initial_flags=cluster_flags if cluster_flags else sim.default_flags,
                    relevant_flag_prefixes=relevant_flag_prefixes,
                )
                # Annotate cluster and source for traceability
                for ex in qa_examples:
                    ex['cluster_key'] = cluster_key
                    ex['source_json'] = json_path
                all_qa.extend(qa_examples)

            # Update cluster flags by unioning flags set True and removing flags set False across any path
            # This provides a conservative context signal without exploding combinations
            if approval_paths:
                flags_true, flags_false = _collect_flag_effects_from_paths(sim, approval_paths)
                cluster_flags.update(flags_true)
                for rem in flags_false:
                    if rem in cluster_flags:
                        cluster_flags.remove(rem)

    # Export aggregated QA dataset
    output_dir = os.path.dirname(export_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(export_file, 'w', encoding='utf-8') as f_out:
        json.dump(all_qa, f_out, indent=2, ensure_ascii=False)
    print(f"{Fore.GREEN}Cluster QA dataset exported: {export_file} (examples: {len(all_qa)}){Style.RESET_ALL}")
    return export_file

    

def main():
    print(f"{Fore.CYAN}Baldur's Gate 3 Dialog Simulator{Style.RESET_ALL}")
    print("This tool allows you to explore the dialog trees from the game.")
    print(f"{Fore.GREEN}TRAVERSAL MODE: Complete dialog tree traversal{Style.RESET_ALL}")
    print(f"- Displays and traverses all child nodes")
    print(f"- Automatically follows jump nodes to their destinations")
    print(f"- Follows goto links for nodes without children")
    print(f"- Follows link fields for nodes without children and goto")
    print(f"{Fore.YELLOW}Test mode is available to bypass flag requirements{Style.RESET_ALL}")
    print(f"{Fore.BLUE}Export options: Save dialog paths to text and JSON files{Style.RESET_ALL}")
    
    # Check if the dialog JSON exists
    if not os.path.isfile(DEFAULT_DIALOG_JSON):
        print(f"{Fore.RED}Error: Required dialog JSON not found at '{DEFAULT_DIALOG_JSON}'.{Style.RESET_ALL}")
        print("Please run the parser script first to generate the dialog tree.")
        return
    
    simulator = DialogSimulator(DEFAULT_DIALOG_JSON)
    
    while True:
        print("\nSelect mode:")
        print("1. Interactive Mode - Explore dialogs with choices")
        print("2. Simulation Mode - Analyze all possible dialog paths")
        print("3. Test Specific Node - Start from a specific node ID")
        print("4. Reset state")
        print("5. View companion approval history")
        print("6. Export approval history to JSON")
        print("7. Export paths to Python dictionary")
        print(f"8. {Fore.MAGENTA}Visualize Dialog Structure (Graphviz){Style.RESET_ALL}") # New option
        print("9. Simulate only paths that include approvals")
        print("10. Generate LLM context for approval QA examples")
        print("11. Generate cluster-aware context for this dialogue file (LLM)")
        print("0. Exit")
        
        try:
            choice = int(input("\nEnter choice: "))
            
            if choice == 0:
                break
            elif choice == 1:
                simulator.interactive_mode()
            elif choice == 2:
                print("\nSimulation Options:")
                print(f"{Fore.YELLOW}Note: Simulation traverses all dialog paths including child nodes, jump nodes, goto links, and link fields{Style.RESET_ALL}")
                print("1. Quick simulation (max depth 5)")
                print("2. Full simulation (unlimited depth)")
                print("3. Custom depth simulation")
                
                try:
                    sim_choice = int(input("\nSelect simulation type: "))
                    
                    # Ask if test mode should be enabled
                    test_mode = input(f"\n{Fore.YELLOW}Enable test mode to ignore flag requirements? (y/n, default: n):{Style.RESET_ALL} ").lower() == 'y'
                    
                    # Ask about export options
                    export_txt = input(f"\n{Fore.BLUE}Export dialog paths to text file? (y/n, default: n):{Style.RESET_ALL} ").lower() == 'y'
                    export_json = input(f"\n{Fore.BLUE}Export traversal data to JSON file? (y/n, default: n):{Style.RESET_ALL} ").lower() == 'y'
                    export_dict = input(f"\n{Fore.BLUE}Export paths to Python dictionary? (y/n, default: n):{Style.RESET_ALL} ").lower() == 'y'
                    
                    # Ask about verbose mode
                    verbose = input(f"\n{Fore.BLUE}Enable verbose logging during simulation? (y/n, default: n):{Style.RESET_ALL} ").lower() == 'y'
                    
                    if sim_choice == 1:
                        # Quick simulation with limited depth
                        print("\nRunning quick simulation (max depth 5)...")
                        _, txt_file, json_file, dict_file = simulator.simulate_all_paths(
                            max_depth=5, 
                            test_mode=test_mode,
                            export_txt=export_txt,
                            export_json=export_json,
                            export_dict=export_dict,
                            verbose=verbose
                        )
                        
                        if txt_file or json_file or dict_file:
                            print(f"{Fore.GREEN}Export completed:{Style.RESET_ALL}")
                            if txt_file:
                                print(f"- Text file: {txt_file}")
                            if json_file:
                                print(f"- JSON file: {json_file}")
                            if dict_file:
                                print(f"- Dictionary file: {dict_file}")
                                
                    elif sim_choice == 2:
                        # Full simulation with high max depth to ensure all paths are found
                        print("\nRunning full simulation (this may take a while)...")
                        _, txt_file, json_file, dict_file = simulator.simulate_all_paths(
                            max_depth=50, 
                            test_mode=test_mode,
                            export_txt=export_txt,
                            export_json=export_json,
                            export_dict=export_dict,
                            verbose=verbose
                        )
                        
                        if txt_file or json_file or dict_file:
                            print(f"{Fore.GREEN}Export completed:{Style.RESET_ALL}")
                            if txt_file:
                                print(f"- Text file: {txt_file}")
                            if json_file:
                                print(f"- JSON file: {json_file}")
                            if dict_file:
                                print(f"- Dictionary file: {dict_file}")
                                
                    elif sim_choice == 3:
                        # Custom depth simulation
                        depth = 10  # Default
                        try:
                            depth_input = input("Maximum dialog depth to simulate (default 10): ")
                            if depth_input.strip():
                                depth = int(depth_input)
                        except ValueError:
                            print(f"{Fore.YELLOW}Using default depth of 10.{Style.RESET_ALL}")
                        
                        print_detailed = input("Print all paths? (y/n, default n): ").lower() == 'y'
                        
                        _, txt_file, json_file, dict_file = simulator.simulate_all_paths(
                            max_depth=depth, 
                            print_paths=print_detailed, 
                            test_mode=test_mode,
                            export_txt=export_txt,
                            export_json=export_json,
                            export_dict=export_dict,
                            verbose=verbose
                        )
                        
                        if txt_file or json_file or dict_file:
                            print(f"{Fore.GREEN}Export completed:{Style.RESET_ALL}")
                            if txt_file:
                                print(f"- Text file: {txt_file}")
                            if json_file:
                                print(f"- JSON file: {json_file}")
                            if dict_file:
                                print(f"- Dictionary file: {dict_file}")
                                
                    else:
                        print(f"{Fore.RED}Invalid choice. Returning to main menu.{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Please enter a number.{Style.RESET_ALL}")
            elif choice == 3:
                try:
                    node_id = input("\nEnter node ID to test (e.g., 134): ")
                    
                    if node_id and node_id in simulator.all_nodes:
                        # Ask about export options
                        export_txt = input(f"\n{Fore.BLUE}Export traversal to text file? (y/n, default: n):{Style.RESET_ALL} ").lower() == 'y'
                        export_json = input(f"\n{Fore.BLUE}Export traversal data to JSON file? (y/n, default: n):{Style.RESET_ALL} ").lower() == 'y'
                        export_approval = input(f"\n{Fore.BLUE}Export approval history to JSON file? (y/n, default: n):{Style.RESET_ALL} ").lower() == 'y'
                        
                        # Test mode is automatically enabled in explore_dialog_from_node
                        print(f"\n{Fore.GREEN}Testing node {node_id}... (traversing complete dialog tree with test mode ON){Style.RESET_ALL}")
                        print(f"{Fore.YELLOW}Test mode is enabled - flag requirements will be ignored{Style.RESET_ALL}")
                        
                        # Run the exploration with export options
                        _, txt_file, json_file, approval_file = simulator.explore_dialog_from_node(
                            node_id,
                            export_txt=export_txt,
                            export_json=export_json,
                            export_approval=export_approval
                        )
                        
                        # Show status after traversal
                        simulator.show_companion_status()
                        
                        # Report on exports if any
                        if txt_file or json_file or approval_file:
                            print(f"{Fore.GREEN}Export completed:{Style.RESET_ALL}")
                            if txt_file:
                                print(f"- Text file: {txt_file}")
                            if json_file:
                                print(f"- JSON file: {json_file}")
                            if approval_file:
                                print(f"- Approval history: {approval_file}")
                    else:
                        print(f"{Fore.RED}Invalid node ID. Node {node_id} not found in the dialog tree.{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Please enter a valid node ID.{Style.RESET_ALL}")
            elif choice == 4:
                simulator.reset_state()
            elif choice == 5:
                # Display approval history
                simulator.show_approval_history()
            elif choice == 6:
                # Export approval history to JSON
                if any(len(changes) > 0 for changes in simulator.companion_approval_history.values()):
                    # Customize the output filename
                    filename = input(f"\nEnter filename for export (default: approval_history.json): ")
                    if not filename:
                        filename = "approval_history.json"
                    elif not filename.endswith(".json"):
                        filename += ".json"
                    
                    # Export approval history
                    output_file = simulator.export_approval_history(filename)
                    print(f"{Fore.GREEN}Approval history exported to {output_file}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}No approval changes to export. Try exploring some dialogs first.{Style.RESET_ALL}")
            elif choice == 7:
                # Export paths to Python dictionary
                print("\nSimulation Options:")
                print(f"{Fore.YELLOW}Note: We need to simulate all paths before exporting to dictionary{Style.RESET_ALL}")
                print("1. Quick simulation (max depth 5)")
                print("2. Full simulation (unlimited depth)")
                print("3. Custom depth simulation")
                
                try:
                    sim_choice = int(input("\nSelect simulation type: "))
                    
                    # Ask if test mode should be enabled
                    test_mode = input(f"\n{Fore.YELLOW}Enable test mode to ignore flag requirements? (y/n, default: n):{Style.RESET_ALL} ").lower() == 'y'
                    
                    # Customize the output filename
                    filename = input(f"\nEnter filename for export (default: dialog_dict.py): ")
                    if not filename:
                        filename = "dialog_dict.py"
                    elif not filename.endswith(".py"):
                        filename += ".py"
                    if not filename.startswith("output_json/"):
                        filename = "output_json/" + filename
                    if sim_choice == 1:
                        # Quick simulation with limited depth
                        print("\nRunning quick simulation (max depth 5)...")
                        all_paths, _, _, _ = simulator.simulate_all_paths(
                            max_depth=5, 
                            test_mode=test_mode,
                            export_txt=False,
                            export_json=False,
                            export_dict=False,
                            verbose=False
                        )
                        
                    elif sim_choice == 2:
                        # Full simulation with high max depth to ensure all paths are found
                        print("\nRunning full simulation (this may take a while)...")
                        all_paths, _, _, _ = simulator.simulate_all_paths(
                            max_depth=50, 
                            test_mode=test_mode,
                            export_txt=False,
                            export_json=False,
                            export_dict=False,
                            verbose=False
                        )
                        
                    elif sim_choice == 3:
                        # Custom depth simulation
                        depth = 10  # Default
                        try:
                            depth_input = input("Maximum dialog depth to simulate (default 10): ")
                            if depth_input.strip():
                                depth = int(depth_input)
                        except ValueError:
                            print(f"{Fore.YELLOW}Using default depth of 10.{Style.RESET_ALL}")
                        
                        all_paths, _, _, _ = simulator.simulate_all_paths(
                            max_depth=depth, 
                            print_paths=False, 
                            test_mode=test_mode,
                            export_txt=False,
                            export_json=False,
                            export_dict=False,
                            verbose=False
                        )
                    else:
                        print(f"{Fore.RED}Invalid choice. Returning to main menu.{Style.RESET_ALL}")
                        continue
                    
                    # Export paths to Python dictionary
                    if all_paths:
                        output_file = simulator.export_paths_to_dict(all_paths, filename)
                        print(f"{Fore.GREEN}Paths exported to {output_file}{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.YELLOW}No paths were generated. Cannot export.{Style.RESET_ALL}")
                        
                except ValueError:
                    print(f"{Fore.RED}Please enter a number.{Style.RESET_ALL}")
            elif choice == 8: # New case for visualization
                if not GRAPHVIZ_AVAILABLE:
                     print(f"{Fore.RED}Graphviz visualization is not available. Please install 'graphviz' and ensure the Graphviz software is in your PATH.{Style.RESET_ALL}")
                     continue

                print(f"\n{Fore.MAGENTA}--- Visualize Dialog Structure ---{Style.RESET_ALL}")
                try:
                    default_filename = f"visualizations/{simulator.metadata.get('synopsis','dialog').replace(' ', '_')}_structure"
                    output_filename = input(f"Enter output base filename (e.g., visualizations/my_dialog_viz) [default: {default_filename}]: ")
                    if not output_filename.strip():
                        output_filename = default_filename

                    start_node_id_input = input("Enter start node ID (leave blank to start from all roots): ")
                    start_node_id = start_node_id_input.strip() if start_node_id_input.strip() else None

                    max_depth_input = input("Enter maximum visualization depth [default: 15]: ")
                    max_depth = int(max_depth_input) if max_depth_input.strip() else 15

                    render_format_input = input("Enter output format (pdf, png, svg, etc.) [default: pdf]: ")
                    render_format = render_format_input.strip().lower() if render_format_input.strip() else 'pdf'

                    # Call the visualization function
                    simulator.visualize_structure(
                        output_filename=output_filename,
                        start_node_id=start_node_id,
                        max_depth=max_depth,
                        render_format=render_format
                    )

                except ValueError:
                    print(f"{Fore.RED}Invalid input (e.g., depth must be a number). Please try again.{Style.RESET_ALL}")
                except Exception as e:
                     print(f"{Fore.RED}An unexpected error occurred during visualization setup: {e}{Style.RESET_ALL}")
            elif choice == 9:
                # Approval-only simulation
                print("\nSimulation Options (Approval Paths):")
                print(f"{Fore.YELLOW}Note: Simulation traverses all dialog paths including child nodes, jump nodes, goto links, and link fields, then filters to those with approvals{Style.RESET_ALL}")
                print("1. Quick simulation (max depth 5)")
                print("2. Full simulation (unlimited depth)")
                print("3. Custom depth simulation")

                try:
                    sim_choice = int(input("\nSelect simulation type: "))

                    # Ask if test mode should be enabled
                    test_mode = input(f"\n{Fore.YELLOW}Enable test mode to ignore flag requirements? (y/n, default: n):{Style.RESET_ALL} ").lower() == 'y'

                    # Ask about export options
                    export_txt = input(f"\n{Fore.BLUE}Export dialog paths to text file? (y/n, default: n):{Style.RESET_ALL} ").lower() == 'y'
                    export_json = input(f"\n{Fore.BLUE}Export traversal data to JSON file? (y/n, default: n):{Style.RESET_ALL} ").lower() == 'y'
                    export_dict = input(f"\n{Fore.BLUE}Export paths to Python dictionary? (y/n, default: n):{Style.RESET_ALL} ").lower() == 'y'

                    # Ask about verbose mode
                    verbose = input(f"\n{Fore.BLUE}Enable verbose logging during simulation? (y/n, default: n):{Style.RESET_ALL} ").lower() == 'y'

                    if sim_choice == 1:
                        print("\nRunning quick approval-path simulation (max depth 5)...")
                        _, txt_file, json_file, dict_file = simulator.simulate_approval_paths(
                            max_depth=5,
                            test_mode=test_mode,
                            export_txt=export_txt,
                            export_json=export_json,
                            export_dict=export_dict,
                            verbose=verbose
                        )
                    elif sim_choice == 2:
                        print("\nRunning full approval-path simulation (this may take a while)...")
                        _, txt_file, json_file, dict_file = simulator.simulate_approval_paths(
                            max_depth=50,
                            test_mode=test_mode,
                            export_txt=export_txt,
                            export_json=export_json,
                            export_dict=export_dict,
                            verbose=verbose
                        )
                    elif sim_choice == 3:
                        depth = 10
                        try:
                            depth_input = input("Maximum dialog depth to simulate (default 10): ")
                            if depth_input.strip():
                                depth = int(depth_input)
                        except ValueError:
                            print(f"{Fore.YELLOW}Using default depth of 10.{Style.RESET_ALL}")
                        print_detailed = input("Print all paths? (y/n, default n): ").lower() == 'y'
                        _, txt_file, json_file, dict_file = simulator.simulate_approval_paths(
                            max_depth=depth,
                            print_paths=print_detailed,
                            test_mode=test_mode,
                            export_txt=export_txt,
                            export_json=export_json,
                            export_dict=export_dict,
                            verbose=verbose
                        )
                    else:
                        print(f"{Fore.RED}Invalid choice. Returning to main menu.{Style.RESET_ALL}")
                        continue

                    if txt_file or json_file or dict_file:
                        print(f"{Fore.GREEN}Export completed:{Style.RESET_ALL}")
                        if txt_file:
                            print(f"- Text file: {txt_file}")
                        if json_file:
                            print(f"- JSON file: {json_file}")
                        if dict_file:
                            print(f"- Dictionary file: {dict_file}")
                except ValueError:
                    print(f"{Fore.RED}Please enter a number.{Style.RESET_ALL}")
            elif choice == 10:
                # LLM context generation for approval QA examples (one context per raw file)
                try:
                    print(f"\n{Fore.MAGENTA}--- Generate LLM Context for Approval QA ---{Style.RESET_ALL}")
                    model_input = input("Enter LiteLLM model id [default: openai/gpt-5-mini] (e.g., openai/gpt-5-mini or gemini/gemini-2.5-flash): ").strip()
                    model = model_input if model_input else 'openai/gpt-5-mini'

                    depth_input = input("Maximum traversal depth [default: 50]: ").strip()
                    max_depth = int(depth_input) if depth_input else 50

                    test_mode = input(f"Enable test mode to ignore flag requirements? (y/n, default: y): ").lower() != 'n'

                    prefixes_input = input("Relevant flag prefixes (comma-separated, optional): ").strip()
                    relevant_flag_prefixes = [p.strip() for p in prefixes_input.split(',') if p.strip()] if prefixes_input else None

                    limit_input = input("Maximum number of QA examples to print [default: 3]: ").strip()
                    limit = int(limit_input) if limit_input else 3

                    temp_input = input("LLM temperature [default: 0.2]: ").strip()
                    temperature = float(temp_input) if temp_input else 0.2

                    mtok_input = input("LLM max tokens [default: 8000]: ").strip()
                    max_tokens = int(mtok_input) if mtok_input else 8000

                    # Discover approval paths
                    approval_paths, _, _, _ = simulator.simulate_approval_paths(
                        max_depth=max_depth,
                        print_paths=False,
                        test_mode=test_mode,
                        export_txt=False,
                        export_json=False,
                        export_dict=False,
                        verbose=False
                    )

                    if not approval_paths:
                        print(f"{Fore.YELLOW}No approval paths found in this dialogue.{Style.RESET_ALL}")
                        continue

                    # Build all QA examples for this file
                    all_examples = []
                    for path in approval_paths:
                        exs = simulator.build_qa_examples_from_path(
                            path,
                            initial_flags=simulator.default_flags,
                            relevant_flag_prefixes=relevant_flag_prefixes
                        )
                        all_examples.extend(exs)

                    if not all_examples:
                        print(f"{Fore.YELLOW}No approval QA examples produced.{Style.RESET_ALL}")
                        continue

                    # Choose canonical dialogue turns and also collect top-3 samples
                    canonical = max(all_examples, key=lambda e: len(e.get('context_dialogue', [])))
                    canonical_turns = canonical.get('context_dialogue', [])
                    top_examples = sorted(
                        all_examples,
                        key=lambda e: len(e.get('context_dialogue', [])),
                        reverse=True
                    )[:3]
                    top3_samples = [ex.get('context_dialogue', []) for ex in top_examples]

                    # Aggregate active flags across examples (filtered already if prefixes provided)
                    aggregated_flags = set()
                    for ex in all_examples:
                        for fl in ex.get('active_flags', []):
                            aggregated_flags.add(fl)
                    aggregated_flags_list = sorted(aggregated_flags)

                    # Generate one LLM context for the entire raw file using up to 3 samples
                    synopsis = simulator.metadata.get('synopsis', '')
                    cluster_synopses = simulator._gather_cluster_synopses(
                        cluster_index_file='goals_to_json_paths.json',
                        include_current=False,
                        include_prior=True,
                        include_future=False,
                        limit=10,
                        use_context_outputs=True,
                        contexts_root='qa-contexts'
                    )

                    llm_ctx_for_file = simulator.generate_context_with_llm(
                        synopsis=synopsis,
                        dialogue_samples=top3_samples if top3_samples else [canonical_turns],
                        active_flags=None,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        cluster_synopses=cluster_synopses
                    )

                    # Print up to 'limit' QA examples, reusing the single context
                    print(f"\n{Fore.CYAN}Generated one shared context for this dialogue file using model: {model}{Style.RESET_ALL}")
                    printed = 0
                    for ex in all_examples:
                        if printed >= limit:
                            break

                        dialogue_turns = ex.get('context_dialogue', [])
                        active_flags = ex.get('active_flags', [])

                        print("\n================= QA EXAMPLE =================")
                        print(f"Approval Node: {ex.get('approval_node_id', '')}")
                        print(f"Active Flags ({len(active_flags)}): {', '.join(active_flags) if active_flags else 'None'}")
                        print("\nContext (LLM - shared for this file):")
                        print(llm_ctx_for_file or "[No LLM output]")
                        print("\nDialogue Window:")
                        for t in dialogue_turns:
                            sp = t.get('speaker', 'Unknown')
                            tx = t.get('text', '')
                            cx = t.get('context', '')
                            nid = t.get('node_id', '')
                            line = f"- {sp}: {tx} (node {nid})"
                            if cx:
                                line += f" || [context] {cx}"
                            print(line)
                        print("==============================================\n")

                        printed += 1

                except ValueError:
                    print(f"{Fore.RED}Invalid input. Returning to main menu.{Style.RESET_ALL}")
            elif choice == 11:
                # Cluster-aware single-context generation for the current file
                try:
                    print(f"\n{Fore.MAGENTA}--- Generate Cluster-Aware Context ---{Style.RESET_ALL}")
                    model_input = input("Enter LiteLLM model id [default: openai/gpt-5-mini]: ").strip()
                    model = model_input if model_input else 'openai/gpt-5-mini'

                    depth_input = input("Maximum traversal depth [default: 50]: ").strip()
                    max_depth = int(depth_input) if depth_input else 50

                    test_mode = input(f"Enable test mode to ignore flag requirements? (y/n, default: y): ").lower() != 'n'

                    prefixes_input = input("Relevant flag prefixes (comma-separated, optional): ").strip()
                    relevant_flag_prefixes = [p.strip() for p in prefixes_input.split(',') if p.strip()] if prefixes_input else None

                    temp_input = input("LLM temperature [default: 0.2]: ").strip()
                    temperature = float(temp_input) if temp_input else 0.2

                    mtok_input = input("LLM max tokens [default: 8000]: ").strip()
                    max_tokens = int(mtok_input) if mtok_input else 8000

                    cluster_file_input = input("Cluster index path [default: goals_to_json_paths.json]: ").strip()
                    cluster_index_file = cluster_file_input if cluster_file_input else 'goals_to_json_paths.json'

                    llm_ctx = simulator.generate_cluster_context_for_current_file(
                        cluster_index_file=cluster_index_file,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        relevant_flag_prefixes=relevant_flag_prefixes,
                        max_depth=max_depth,
                        test_mode=test_mode,
                        verbose=False
                    )

                    print("\n================= FILE CONTEXT =================")
                    print(llm_ctx or "[No LLM output]")
                    print("===============================================\n")

                except ValueError:
                    print(f"{Fore.RED}Invalid input. Returning to main menu.{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Invalid choice. Try again.{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Please enter a number.{Style.RESET_ALL}")

if __name__ == "__main__":
    # Add creation of visualization directory if it doesn't exist
    if not os.path.exists('visualizations'):
        try:
            os.makedirs('visualizations')
            print(f"{Fore.GREEN}Created directory 'visualizations' for graph output.{Style.RESET_ALL}")
        except OSError as e:
            print(f"{Fore.RED}Could not create directory 'visualizations': {e}{Style.RESET_ALL}")
    main()