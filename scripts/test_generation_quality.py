#!/usr/bin/env python3
"""
Quick test generation script for quality checking
Generates small samples with both constitutions
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    """Run quick test generation"""
    
    print("="*60)
    print("Constitutional AI - Quick Test Generation")
    print("="*60)
    print()
    
    # Configuration
    HM7B_PATH = project_root.parent / "Constitutional_AI_Project" / "trained_models" / "hm7b"
    DATA_DIR = project_root / "data"
    OUTPUT_DIR = DATA_DIR / "sl_datasets" / "test"
    
    # Create test output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Test parameters - SMALL for quality checking
    NUM_RED_TEAM = 5
    NUM_HELPFUL = 5
    NUM_REVISIONS = 4  # Keep full revisions to test quality
    
    # Check HM7B exists
    if not HM7B_PATH.exists():
        print(f"❌ HM7B model not found at {HM7B_PATH}")
        print("Please ensure the v1 project is in the parent directory")
        sys.exit(1)
    
    print(f"✅ Found HM7B model at {HM7B_PATH}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📊 Samples: {NUM_RED_TEAM} red-team, {NUM_HELPFUL} helpful")
    print(f"🔄 Revisions: {NUM_REVISIONS}")
    print()
    
    # Track timing
    start_time = time.time()
    
    # Generate Deontological test dataset
    print("="*60)
    print("1️⃣ DEONTOLOGICAL DATASET")
    print("="*60)
    
    deont_cmd = f"""python "{project_root}/scripts/generate_sl_cai_dataset.py" \
        --constitution deontological \
        --constitution-path "{project_root}/constitutions/deontological/principles.json" \
        --model "{HM7B_PATH}" \
        --red-team-path "{DATA_DIR}/red_team/sample_red_team.json" \
        --helpful-path "{DATA_DIR}/helpfulness/sample_helpful.json" \
        --output-dir "{OUTPUT_DIR}" \
        --num-red-team {NUM_RED_TEAM} \
        --num-helpful {NUM_HELPFUL} \
        --num-revisions {NUM_REVISIONS} \
        --format jsonl"""
    
    print(f"Running: {deont_cmd}\n")
    deont_start = time.time()
    
    result = os.system(deont_cmd)
    
    deont_time = time.time() - deont_start
    
    if result == 0:
        print(f"\n✅ Deontological dataset generated in {deont_time:.1f} seconds")
    else:
        print(f"\n❌ Failed to generate deontological dataset")
        sys.exit(1)
    
    # Generate Consequentialist test dataset
    print("\n" + "="*60)
    print("2️⃣ CONSEQUENTIALIST DATASET")
    print("="*60)
    
    conseq_cmd = f"""python "{project_root}/scripts/generate_sl_cai_dataset.py" \
        --constitution consequentialist \
        --constitution-path "{project_root}/constitutions/consequentialist/principles.json" \
        --model "{HM7B_PATH}" \
        --red-team-path "{DATA_DIR}/red_team/sample_red_team.json" \
        --helpful-path "{DATA_DIR}/helpfulness/sample_helpful.json" \
        --output-dir "{OUTPUT_DIR}" \
        --num-red-team {NUM_RED_TEAM} \
        --num-helpful {NUM_HELPFUL} \
        --num-revisions {NUM_REVISIONS} \
        --format jsonl"""
    
    print(f"Running: {conseq_cmd}\n")
    conseq_start = time.time()
    
    result = os.system(conseq_cmd)
    
    conseq_time = time.time() - conseq_start
    
    if result == 0:
        print(f"\n✅ Consequentialist dataset generated in {conseq_time:.1f} seconds")
    else:
        print(f"\n❌ Failed to generate consequentialist dataset")
        sys.exit(1)
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "="*60)
    print("📊 GENERATION COMPLETE")
    print("="*60)
    
    # List generated files
    print("\n📁 Generated files:")
    for file in OUTPUT_DIR.glob("*.jsonl"):
        size = file.stat().st_size / 1024  # KB
        print(f"  - {file.name}: {size:.1f} KB")
    
    # Time analysis
    print(f"\n⏱️ Timing:")
    print(f"  - Deontological: {deont_time:.1f} seconds")
    print(f"  - Consequentialist: {conseq_time:.1f} seconds")
    print(f"  - Total: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    # Projection for full dataset
    samples_generated = (NUM_RED_TEAM + NUM_HELPFUL) * 2  # Both constitutions
    full_samples = 200 * 2  # 100 red + 100 helpful, both constitutions
    projected_time = (total_time / samples_generated) * full_samples
    
    print(f"\n📈 Projection for full dataset (100 samples each):")
    print(f"  - Estimated time: {projected_time/60:.0f} minutes ({projected_time/3600:.1f} hours)")
    
    print(f"\n✨ Test datasets ready for quality analysis!")
    print(f"   Run: python scripts/analyze_dataset_quality.py")
    
    # Save generation metadata
    metadata = {
        "generation_date": datetime.now().isoformat(),
        "model": str(HM7B_PATH),
        "num_red_team": NUM_RED_TEAM,
        "num_helpful": NUM_HELPFUL,
        "num_revisions": NUM_REVISIONS,
        "deont_time_seconds": deont_time,
        "conseq_time_seconds": conseq_time,
        "total_time_seconds": total_time,
        "projected_full_time_minutes": projected_time / 60
    }
    
    with open(OUTPUT_DIR / "generation_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()