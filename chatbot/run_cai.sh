#!/bin/bash

echo "Constitutional AI Chat Launcher"
echo "=============================="
echo ""
echo "Select mode:"
echo "1) Deontological (duty-based ethics)"
echo "2) Consequentialist (outcome-based ethics)"  
echo "3) Compare (run both side-by-side)"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo "Starting Deontological model..."
        python3 cai_chat.py --mode deont
        ;;
    2)
        echo "Starting Consequentialist model..."
        python3 cai_chat.py --mode conseq
        ;;
    3)
        echo "Starting Compare mode..."
        python3 cai_chat.py --mode compare
        ;;
    *)
        echo "Invalid choice. Please run again and select 1, 2, or 3."
        exit 1
        ;;
esac