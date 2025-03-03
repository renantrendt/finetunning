#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Command Line Interface for the Yanomami RAG Translator
---------------------------------------------------------
An interactive terminal interface for the Yanomami-English RAG translation system.
"""

import os
import sys
import time
from yanomami_rag_translator import YanomamiRAGTranslator

class Colors:
    """Terminal colors."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def clear_screen():
    """Clears the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Prints the application header."""
    clear_screen()
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'Yanomami - translator':^60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")

def print_menu():
    """Prints the main menu."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}Menu:{Colors.ENDC}")
    print(f"{Colors.BLUE}1. Simple Translation{Colors.ENDC}")
    print(f"{Colors.BLUE}2. Comprehensive Query{Colors.ENDC}")
    print(f"{Colors.BLUE}3. Examples{Colors.ENDC}")
    print(f"{Colors.BLUE}4. Exit{Colors.ENDC}")
    return input(f"\n{Colors.BOLD}Choose an option (1-4): {Colors.ENDC}")

def translate_query(translator):
    """Interface for translating a simple query."""
    clear_screen()
    print(f"{Colors.BOLD}{Colors.GREEN}Simple Translation{Colors.ENDC}\n")
    print("Enter your query below. Format examples:")
    print("- Translate to Yanomami: They are not very far away")
    print("- Translate to English: pë ahetoimi kë thë")
    print(f"\n{Colors.YELLOW}Type 'back' to return to the main menu.{Colors.ENDC}")
    
    while True:
        query = input(f"\n{Colors.BOLD}Query: {Colors.ENDC}")
        if query.lower() in ['back', 'return']:
            return
        
        if not query.strip():
            print(f"{Colors.RED}Please enter a valid query.{Colors.ENDC}")
            continue
        
        print(f"\n{Colors.YELLOW}Researching and generating translation...{Colors.ENDC}")
        start_time = time.time()
        
        try:
            result = translator.translate(query)
            elapsed_time = time.time() - start_time
            
            print(f"\n{Colors.GREEN}{Colors.BOLD}Result:{Colors.ENDC}")
            print(f"{result}")
            print(f"\n{Colors.BLUE}Time: {elapsed_time:.2f} seconds{Colors.ENDC}")
        except Exception as e:
            print(f"\n{Colors.RED}Error translating: {str(e)}{Colors.ENDC}")
        
        input(f"\n{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")
        clear_screen()
        print(f"{Colors.BOLD}{Colors.GREEN}Simple Translation{Colors.ENDC}\n")
        print("Enter your query below or 'back' to return to the main menu:")

def comprehensive_query(translator):
    """Interface for comprehensive query about a word or phrase."""
    clear_screen()
    print(f"{Colors.BOLD}{Colors.GREEN}Comprehensive Query{Colors.ENDC}\n")
    print("Enter a word or phrase to get comprehensive information:")
    print("- Examples: 'ahetoimi', 'hello', 'pë ahetoimi kë thë'")
    print(f"\n{Colors.YELLOW}Type 'back' to return to the main menu.{Colors.ENDC}")
    
    while True:
        query = input(f"\n{Colors.BOLD}Word or phrase: {Colors.ENDC}")
        if query.lower() in ['back', 'return']:
            return
        
        if not query.strip():
            print(f"{Colors.RED}Please enter a valid word or phrase.{Colors.ENDC}")
            continue
        
        print(f"\n{Colors.YELLOW}Searching for comprehensive information...{Colors.ENDC}")
        start_time = time.time()
        
        try:
            result = translator.translate(query, comprehensive=True)
            elapsed_time = time.time() - start_time
            
            print(f"\n{Colors.GREEN}{Colors.BOLD}Comprehensive Information:{Colors.ENDC}")
            print(f"{result}")
            print(f"\n{Colors.BLUE}Time: {elapsed_time:.2f} seconds{Colors.ENDC}")
        except Exception as e:
            print(f"\n{Colors.RED}Error retrieving information: {str(e)}{Colors.ENDC}")
        
        input(f"\n{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")
        clear_screen()
        print(f"{Colors.BOLD}{Colors.GREEN}Comprehensive Query{Colors.ENDC}\n")
        print("Enter a word or phrase or 'back' to return to the main menu:")

def use_examples(translator):
    """Interface for using predefined examples."""
    examples = [
        ("Simple: Meaning of 'ahetoimi'", "What does 'ahetoimi' mean in Yanomami?"),
        ("Simple: Meaning of 'aheprariyo'", "What does 'aheprariyo' mean in Yanomami?"),
        ("Simple: Translate to Yanomami", "Translate to Yanomami: They are not very far away"),
        ("Simple: Translate to English", "Translate to English: pë ahetoimi kë thë"),
        ("Simple: How to say 'hello'", "How do you say 'hello' in Yanomami?"),
        ("Comprehensive: 'ahetoimi'", "ahetoimi", True),
        ("Comprehensive: 'hello'", "hello", True),
        ("Comprehensive: 'pë'", "pë", True),
        ("Back to main menu", "back")
    ]
    
    while True:
        clear_screen()
        print(f"{Colors.BOLD}{Colors.GREEN}Examples{Colors.ENDC}\n")
        print("Choose an example to try:")
        
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example[0]}")
        
        choice = input(f"\n{Colors.BOLD}Choose an option (1-{len(examples)}): {Colors.ENDC}")
        
        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(examples):
                raise ValueError("Invalid option")
            
            example = examples[idx]
            desc = example[0]
            query = example[1]
            
            if query == "back":
                return
            
            # Check if it's a comprehensive query
            is_comprehensive = len(example) > 2 and example[2]
            
            clear_screen()
            print(f"{Colors.BOLD}{Colors.GREEN}Example: {desc}{Colors.ENDC}\n")
            print(f"{Colors.BOLD}Query: {query}{Colors.ENDC}")
            
            print(f"\n{Colors.YELLOW}Processing...{Colors.ENDC}")
            start_time = time.time()
            
            try:
                if is_comprehensive:
                    result = translator.translate(query, comprehensive=True)
                    print(f"\n{Colors.GREEN}{Colors.BOLD}Comprehensive Information:{Colors.ENDC}")
                else:
                    result = translator.translate(query)
                    print(f"\n{Colors.GREEN}{Colors.BOLD}Result:{Colors.ENDC}")
                    
                elapsed_time = time.time() - start_time
                print(f"{result}")
                print(f"\n{Colors.BLUE}Time: {elapsed_time:.2f} seconds{Colors.ENDC}")
            except Exception as e:
                print(f"\n{Colors.RED}Error: {str(e)}{Colors.ENDC}")
            
            input(f"\n{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")
        
        except (ValueError, IndexError):
            print(f"\n{Colors.RED}Invalid option. Press Enter to continue...{Colors.ENDC}")
            input()

def configure_directories():
    """Interface for configuring directories."""
    clear_screen()
    print(f"{Colors.BOLD}{Colors.GREEN}Configure Directories{Colors.ENDC}\n")
    
    dataset_dir = input(f"{Colors.BOLD}Dataset Directory [yanomami_dataset]: {Colors.ENDC}")
    if not dataset_dir:
        dataset_dir = "yanomami_dataset"
    
    model_dir = input(f"{Colors.BOLD}Model Directory [gpt2_yanomami_translator]: {Colors.ENDC}")
    if not model_dir:
        model_dir = "gpt2_yanomami_translator"
    
    # Check if directories exist
    if not os.path.exists(dataset_dir):
        print(f"\n{Colors.RED}Error: Dataset directory not found: {dataset_dir}{Colors.ENDC}")
        input(f"\n{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")
        return None, None
    
    if not os.path.exists(model_dir):
        print(f"\n{Colors.RED}Error: Model directory not found: {model_dir}{Colors.ENDC}")
        input(f"\n{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")
        return None, None
    
    print(f"\n{Colors.GREEN}Directories configured successfully!{Colors.ENDC}")
    input(f"\n{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")
    return dataset_dir, model_dir

def main():
    """Main function."""
    print_header()
    
    # Default configurations
    dataset_dir = "yanomami_dataset"
    model_dir = "gpt2_yanomami_translator"
    
    # Check if directories exist
    if not os.path.exists(dataset_dir):
        print(f"{Colors.RED}Warning: Default dataset directory not found: {dataset_dir}{Colors.ENDC}")
        print(f"{Colors.YELLOW}Please configure the correct directories.{Colors.ENDC}")
        new_dataset_dir, new_model_dir = configure_directories()
        if new_dataset_dir and new_model_dir:
            dataset_dir, model_dir = new_dataset_dir, new_model_dir
        else:
            print(f"{Colors.RED}Could not configure directories. Exiting...{Colors.ENDC}")
            return
    
    if not os.path.exists(model_dir):
        print(f"{Colors.RED}Warning: Default model directory not found: {model_dir}{Colors.ENDC}")
        print(f"{Colors.YELLOW}Please configure the correct directories.{Colors.ENDC}")
        new_dataset_dir, new_model_dir = configure_directories()
        if new_dataset_dir and new_model_dir:
            dataset_dir, model_dir = new_dataset_dir, new_model_dir
        else:
            print(f"{Colors.RED}Could not configure directories. Exiting...{Colors.ENDC}")
            return
    
    # Initialize the translator
    print(f"\n{Colors.YELLOW}Initializing the RAG translator...{Colors.ENDC}")
    try:
        translator = YanomamiRAGTranslator(
            dataset_dir=dataset_dir,
            gpt2_model_dir=model_dir
        )
        print(f"{Colors.GREEN}Translator initialized successfully!{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.RED}Error initializing translator: {str(e)}{Colors.ENDC}")
        input(f"\n{Colors.BOLD}Press Enter to exit...{Colors.ENDC}")
        return
    
    # Main loop
    while True:
        choice = print_menu()
        
        if choice == '1':
            translate_query(translator)
        elif choice == '2':
            comprehensive_query(translator)
        elif choice == '3':
            use_examples(translator)
        elif choice == '4':
            print(f"\n{Colors.GREEN}Thank you for using the Yanomami RAG Translator!{Colors.ENDC}")
            break
        else:
            print(f"\n{Colors.RED}Invalid option. Please choose a valid option.{Colors.ENDC}")
            input(f"\n{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Program interrupted by the user.{Colors.ENDC}")
        print(f"{Colors.GREEN}Thank you for using the Yanomami RAG Translator!{Colors.ENDC}")
    except Exception as e:
        print(f"\n\n{Colors.RED}Unexpected error: {str(e)}{Colors.ENDC}")
        print(f"{Colors.RED}Please restart the program.{Colors.ENDC}")
