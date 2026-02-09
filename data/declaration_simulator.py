#!/usr/bin/env python3
"""
Generate synthetic customs declarations metadata for post-processing validation.
NOTE: Declarations are NOT used in VLM training - only for post-processing testing.
The VLM focuses purely on item recognition; declaration comparison happens in post-processing.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any


# Common legitimate baggage items
LEGITIMATE_ITEMS = [
    "clothing",
    "shoes",
    "toiletries",
    "books",
    "electronics",
    "laptop",
    "tablet",
    "phone charger",
    "headphones",
    "camera",
    "sunglasses",
    "umbrella",
    "medicine",
    "cosmetics",
    "jewelry",
    "watch",
    "wallet",
    "travel documents",
    "snacks",
    "water bottle",
    "towels",
    "travel pillow",
    "portable battery",
]

# Prohibited items (matching OPIXray categories)
PROHIBITED_ITEMS = [
    "folding knife",
    "straight knife",
    "scissors",
    "utility knife",
    "multi-tool knife",
]

# Friendly names for categories
ITEM_MAPPING = {
    "Folding_Knife": "folding knife",
    "Straight_Knife": "straight knife",
    "Scissor": "scissors",
    "Utility_Knife": "utility knife",
    "Multi-tool_Knife": "multi-tool knife",
}


def generate_legitimate_declaration(num_items: int = None) -> List[str]:
    """Generate a realistic legitimate declaration."""
    if num_items is None:
        num_items = random.randint(3, 8)
    
    return random.sample(LEGITIMATE_ITEMS, min(num_items, len(LEGITIMATE_ITEMS)))


def has_prohibited_items(metadata: Dict) -> List[str]:
    """Extract prohibited items from metadata."""
    categories = metadata.get("categories", [])
    prohibited = []
    
    for cat in categories:
        if cat in ITEM_MAPPING:
            prohibited.append(ITEM_MAPPING[cat])
    
    return list(set(prohibited))  # Remove duplicates


def create_matching_declaration(prohibited: List[str]) -> List[str]:
    """Create a declaration that matches (includes prohibited items)."""
    declaration = generate_legitimate_declaration(num_items=random.randint(3, 6))
    
    # Truthfully declare prohibited items (honest traveler scenario)
    for item in prohibited:
        # Sometimes use synonyms or vague terms
        if "knife" in item and random.random() < 0.3:
            declaration.append("pocket knife")
        elif "scissors" in item and random.random() < 0.3:
            declaration.append("cutting tool")
        else:
            declaration.append(item)
    
    return declaration


def create_mismatching_declaration(prohibited: List[str]) -> List[str]:
    """Create a declaration that doesn't match (fraud scenario)."""
    # Just legitimate items, omitting prohibited ones
    return generate_legitimate_declaration(num_items=random.randint(4, 7))


def create_partial_match_declaration(prohibited: List[str]) -> List[str]:
    """Create a declaration that partially matches (some items declared, some hidden)."""
    declaration = generate_legitimate_declaration(num_items=random.randint(3, 5))
    
    # Declare only some prohibited items
    if len(prohibited) > 1:
        num_to_declare = random.randint(1, len(prohibited) - 1)
        declared_prohibited = random.sample(prohibited, num_to_declare)
        declaration.extend(declared_prohibited)
    
    return declaration


def augment_with_declaration(vqa_pair: Dict[str, Any], match_ratio: float = 0.5) -> Dict[str, Any]:
    """
    Add declaration information to metadata ONLY (not for VQA training).
    This metadata can be used later for post-processing validation/testing.
    
    NOTE: We do NOT generate declaration comparison questions anymore.
    The VLM focuses purely on item recognition.
    """
    metadata = vqa_pair.get("metadata", {})
    prohibited = has_prohibited_items(metadata)
    
    # Determine if declaration should match
    if not prohibited:
        # No prohibited items - always use legitimate declaration
        declaration = generate_legitimate_declaration()
        match_declaration = True
    else:
        # Has prohibited items - create match/mismatch based on ratio
        rand = random.random()
        
        if rand < match_ratio:
            # Matching declaration (honest traveler)
            declaration = create_matching_declaration(prohibited)
            match_declaration = True
        elif rand < match_ratio + 0.3:
            # Partial match (some items hidden)
            declaration = create_partial_match_declaration(prohibited)
            match_declaration = False
        else:
            # Complete mismatch (fraud)
            declaration = create_mismatching_declaration(prohibited)
            match_declaration = False
    
    # Add declaration to metadata (for post-processing testing only)
    metadata["declared_items"] = declaration
    metadata["match_declaration"] = match_declaration
    vqa_pair["metadata"] = metadata
    
    # Return only the original VQA pair (no comparison question)
    return vqa_pair


def process_vqa_file(
    input_file: Path,
    output_file: Path,
    match_ratio: float = 0.5,
):
    """
    Add declaration metadata to VQA file (for post-processing testing only).
    Does NOT add declaration comparison questions to training data.
    """
    print(f"Processing {input_file}...")
    print("Note: Adding declaration metadata for post-processing validation only.")
    print("      The VLM will NOT be trained on declaration comparison.")
    
    augmented_pairs = []
    
    with open(input_file, "r") as f:
        for line in f:
            vqa_pair = json.loads(line.strip())
            
            # Augment with declaration metadata only
            augmented_pair = augment_with_declaration(vqa_pair, match_ratio)
            augmented_pairs.append(augmented_pair)
    
    # Write augmented data
    print(f"Writing {len(augmented_pairs)} VQA pairs to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        for pair in augmented_pairs:
            f.write(json.dumps(pair) + "\n")
    
    print(f"✓ Added declaration metadata to {len(augmented_pairs)} VQA pairs")
    
    # Statistics
    with_prohibited = sum(1 for p in augmented_pairs if has_prohibited_items(p.get("metadata", {})))
    mismatches = sum(1 for p in augmented_pairs if not p.get("metadata", {}).get("match_declaration", True))
    
    print(f"  - {with_prohibited} pairs with prohibited items")
    print(f"  - {mismatches} declaration mismatches in metadata")
    print(f"  - {len(augmented_pairs) - mismatches} declaration matches in metadata")
    print(f"\nThis metadata can be used to test post-processing logic.")


def main():
    parser = argparse.ArgumentParser(
        description="Add synthetic customs declarations to VQA dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input VQA JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output VQA JSONL file with declarations",
    )
    parser.add_argument(
        "--match-ratio",
        type=float,
        default=0.5,
        help="Ratio of matching declarations (0.0-1.0, default: 0.5)",
    )
    
    args = parser.parse_args()
    
    input_file = Path(args.input)
    output_file = Path(args.output)
    
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        print("Please run: python data/create_vqa_pairs.py first")
        return 1
    
    process_vqa_file(
        input_file,
        output_file,
        args.match_ratio,
    )
    
    print("\n✓ Declaration metadata added!")
    print("\nIMPORTANT:")
    print("  - VLM training focuses on item recognition ONLY")
    print("  - Declaration comparison is handled in post-processing")
    print("  - See: inference/postprocess.py for declaration matching logic")
    print("\nNext step: python data/split_dataset.py or start training directly")


if __name__ == "__main__":
    main()
