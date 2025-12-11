#!/usr/bin/env python3
"""
Python launcher for Constitutional AI Chat
"""

import subprocess
import sys

def main():
    print("Constitutional AI Chat Launcher")
    print("=" * 30)
    print("\nSelect mode:")
    print("1) Deontological (duty-based ethics)")
    print("2) Consequentialist (outcome-based ethics)")
    print("3) Compare (run both side-by-side)")
    print()
    
    try:
        choice = input("Enter choice [1-3]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting...")
        sys.exit(0)
    
    if choice == "1":
        print("\nStarting Deontological model...")
        subprocess.run(["python3", "cai_chat.py", "--mode", "deont"])
    elif choice == "2":
        print("\nStarting Consequentialist model...")
        subprocess.run(["python3", "cai_chat.py", "--mode", "conseq"])
    elif choice == "3":
        print("\nStarting Compare mode...")
        subprocess.run(["python3", "cai_chat.py", "--mode", "compare"])
    else:
        print("Invalid choice. Please run again and select 1, 2, or 3.")
        sys.exit(1)

if __name__ == "__main__":
    main()