#!/usr/bin/env python3
"""
Attempt to recover lost Claude-generated datasets
Check various locations and temporary files
"""

import os
import json
from pathlib import Path
import glob

def search_for_datasets():
    """Search for any JSONL files that might contain our lost data"""
    
    print("🔍 Searching for potential dataset files...")
    
    # Locations to check
    search_paths = [
        "/Users/leeyoungchan/development/AI_LAB/Constitutional_AI_Project_v2",
        "/Users/leeyoungchan/development/AI_LAB",
        "/tmp",
        "/var/folders",  # Mac temp directories
        "~/Downloads",
        "~/Documents"
    ]
    
    found_files = []
    
    for base_path in search_paths:
        base_path = os.path.expanduser(base_path)
        if os.path.exists(base_path):
            # Search for JSONL files
            pattern = os.path.join(base_path, "**/*.jsonl")
            files = glob.glob(pattern, recursive=True)
            
            for file in files:
                # Check if file might be our dataset
                if 'deontological' in file or 'consequentialist' in file:
                    try:
                        # Check file size and sample count
                        size = os.path.getsize(file)
                        with open(file, 'r') as f:
                            count = sum(1 for _ in f)
                        
                        # Get modification time
                        mtime = os.path.getmtime(file)
                        from datetime import datetime
                        mod_time = datetime.fromtimestamp(mtime)
                        
                        found_files.append({
                            'path': file,
                            'size': size,
                            'samples': count,
                            'modified': mod_time
                        })
                        
                        print(f"\n📁 Found: {file}")
                        print(f"   Size: {size:,} bytes")
                        print(f"   Samples: {count}")
                        print(f"   Modified: {mod_time}")
                        
                        # Check if this could be our lost dataset
                        if count == 1000:  # We generated 500+500
                            print("   ⚠️  This might be the lost dataset! (1000 samples)")
                        
                    except Exception as e:
                        print(f"   Error reading {file}: {e}")
    
    # Check for any backup or autosave files
    print("\n🔍 Checking for backup/autosave files...")
    backup_patterns = [
        "*backup*", "*autosave*", "*tmp*", "*temp*", 
        ".*deontological*", ".*consequentialist*"  # Hidden files
    ]
    
    for pattern in backup_patterns:
        for base_path in ["/Users/leeyoungchan/development/AI_LAB/Constitutional_AI_Project_v2"]:
            base_path = os.path.expanduser(base_path)
            if os.path.exists(base_path):
                files = glob.glob(os.path.join(base_path, "**", pattern), recursive=True)
                for file in files:
                    if file.endswith(('.jsonl', '.json')):
                        print(f"   Backup candidate: {file}")
    
    return found_files

def check_process_output():
    """Check if there's any cached process output"""
    print("\n🔍 Checking for cached process output...")
    
    # Check common log locations
    log_locations = [
        "*.log",
        "nohup.out",
        "./*.out",
        "dataset_generation.log"
    ]
    
    for pattern in log_locations:
        files = glob.glob(pattern)
        for file in files:
            if os.path.exists(file):
                size = os.path.getsize(file)
                if size > 0:
                    print(f"   Log file: {file} ({size:,} bytes)")
                    # Check if it contains dataset data
                    with open(file, 'r') as f:
                        content = f.read(1000)  # Read first 1000 chars
                        if 'deontological' in content or 'consequentialist' in content:
                            print(f"      ⚠️ Contains constitution references!")

def check_memory_dumps():
    """Check for any memory dumps or crash reports"""
    print("\n🔍 Checking for memory dumps or crash reports...")
    
    crash_locations = [
        "~/Library/Logs/DiagnosticReports",
        "/var/log",
        "core.*"
    ]
    
    for location in crash_locations:
        location = os.path.expanduser(location)
        if os.path.exists(location) or glob.glob(location):
            if os.path.isdir(location):
                files = os.listdir(location)[:5]  # Show first 5
                if files:
                    print(f"   Found in {location}: {files}")
            else:
                for file in glob.glob(location):
                    print(f"   Core dump: {file}")

def analyze_existing_datasets():
    """Analyze the existing datasets we found"""
    print("\n📊 Analyzing existing datasets...")
    
    datasets = [
        "/Users/leeyoungchan/development/AI_LAB/Constitutional_AI_Project_v2/deontological_sl_dataset.jsonl",
        "/Users/leeyoungchan/development/AI_LAB/Constitutional_AI_Project_v2/consequentialist_sl_dataset.jsonl"
    ]
    
    for dataset_path in datasets:
        if os.path.exists(dataset_path):
            print(f"\n📁 {os.path.basename(dataset_path)}:")
            
            with open(dataset_path, 'r') as f:
                samples = [json.loads(line) for line in f]
            
            print(f"   Total samples: {len(samples)}")
            
            # Check creation pattern
            if samples:
                # Check first few prompts to identify generation method
                first_prompts = [s.get('prompt', '')[:50] for s in samples[:3]]
                print(f"   First prompts: {first_prompts}")
                
                # Check if this has Claude's signature
                has_critique = sum(1 for s in samples if s.get('critique_applied', False))
                avg_revisions = sum(len(s.get('revisions', [])) for s in samples) / len(samples) if samples else 0
                
                print(f"   Has critique: {has_critique}")
                print(f"   Avg revisions: {avg_revisions:.1f}")
                
                # Check response patterns
                if samples[0].get('response'):
                    response = samples[0]['response']
                    if 'Article' in response or 'categorical' in response:
                        print("   ✅ Contains constitutional language")
                    else:
                        print("   ❌ Missing constitutional language patterns")

def main():
    print("🚨 Lost Dataset Recovery Tool\n")
    print("Looking for the lost Claude-generated datasets...")
    print("Original generation: 500 red team + 500 helpful prompts each")
    print("Expected files: deontological_sl_dataset.jsonl, consequentialist_sl_dataset.jsonl")
    print("=" * 60)
    
    # Search for datasets
    found_files = search_for_datasets()
    
    # Check process outputs
    check_process_output()
    
    # Check for memory dumps
    check_memory_dumps()
    
    # Analyze what we have
    analyze_existing_datasets()
    
    print("\n" + "=" * 60)
    print("📋 Summary:")
    if found_files:
        print(f"Found {len(found_files)} potential dataset files")
        # Find the most likely candidate
        candidates = [f for f in found_files if f['samples'] == 1000]
        if candidates:
            print("\n⚠️  POTENTIAL MATCH FOUND:")
            for c in candidates:
                print(f"   {c['path']}")
                print(f"   Modified: {c['modified']}")
    else:
        print("❌ No recovery candidates found")
        print("\nThe data appears to be completely lost.")
        print("Recommendation: Use the new robust generator script to regenerate.")

if __name__ == "__main__":
    main()