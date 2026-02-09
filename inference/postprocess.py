#!/usr/bin/env python3
"""
Post-processing for X-ray inspection results.
Handles declaration comparison, risk scoring, and business rules.
"""

from typing import List, Dict, Any, Optional
import re


# Prohibited item categories
PROHIBITED_ITEMS = [
    "folding knife",
    "straight knife",
    "scissors",
    "scissor",
    "utility knife",
    "multi-tool knife",
    "knife",
    "blade",
]

# Risk levels
RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"

# Actions
ACTION_CLEAR = "CLEAR"
ACTION_REVIEW = "REVIEW"
ACTION_PHYSICAL_INSPECTION = "PHYSICAL_INSPECTION"


def extract_items_from_text(text: str) -> List[str]:
    """
    Extract detected items from VLM response text.
    
    Args:
        text: VLM generated answer text
    
    Returns:
        List of detected item names
    """
    text_lower = text.lower()
    detected = []
    
    # Check for each prohibited item
    for item in PROHIBITED_ITEMS:
        if item in text_lower:
            detected.append(item)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_items = []
    for item in detected:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)
    
    return unique_items


def normalize_item_name(item: str) -> str:
    """Normalize item name for comparison."""
    item_lower = item.lower().strip()
    
    # Map variations to standard names
    mappings = {
        "scissor": "scissors",
        "pocket knife": "folding knife",
        "blade": "knife",
        "cutting tool": "knife",
    }
    
    return mappings.get(item_lower, item_lower)


def compare_with_declaration(
    detected_items: List[str],
    declared_items: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compare detected items with declaration.
    
    Args:
        detected_items: Items detected by VLM
        declared_items: Items declared by passenger
    
    Returns:
        Comparison results dictionary
    """
    if declared_items is None:
        declared_items = []
    
    # Normalize all items
    detected_normalized = [normalize_item_name(item) for item in detected_items]
    declared_normalized = [normalize_item_name(item) for item in declared_items]
    
    # Find matches and mismatches
    declared_set = set(declared_normalized)
    detected_set = set(detected_normalized)
    
    # Items that are declared and detected
    matched_items = list(detected_set & declared_set)
    
    # Items detected but not declared (potential fraud)
    undeclared_items = list(detected_set - declared_set)
    
    # Items declared but not detected (false declaration or missed detection)
    undetected_items = list(declared_set - detected_set)
    
    # Determine if declaration matches
    declaration_match = len(undeclared_items) == 0 and len(detected_items) > 0
    
    return {
        "declaration_match": declaration_match,
        "matched_items": matched_items,
        "undeclared_items": undeclared_items,
        "undetected_items": undetected_items,
        "has_declaration": len(declared_items) > 0,
    }


def assess_risk_level(
    detected_items: List[str],
    declaration_comparison: Optional[Dict] = None,
    has_occlusion: bool = False,
) -> str:
    """
    Assess risk level based on detected items and declaration.
    
    Args:
        detected_items: Items detected by VLM
        declaration_comparison: Results from compare_with_declaration
        has_occlusion: Whether items are concealed
    
    Returns:
        Risk level: "low", "medium", or "high"
    """
    # No items detected
    if not detected_items:
        return RISK_LOW
    
    # Check for prohibited items
    has_prohibited = any(
        any(prohibited in item.lower() for prohibited in ["knife", "blade", "scissors"])
        for item in detected_items
    )
    
    # High risk conditions
    if has_prohibited:
        # Multiple prohibited items
        if len(detected_items) > 2:
            return RISK_HIGH
        
        # Concealed prohibited items
        if has_occlusion:
            return RISK_HIGH
        
        # Undeclared prohibited items (fraud indicator)
        if declaration_comparison and declaration_comparison.get("undeclared_items"):
            return RISK_HIGH
        
        # Single prohibited item, properly declared
        if declaration_comparison and declaration_comparison.get("declaration_match"):
            return RISK_MEDIUM
        
        # Single prohibited item, no declaration provided
        return RISK_MEDIUM
    
    # No prohibited items
    return RISK_LOW


def determine_action(risk_level: str) -> str:
    """
    Determine recommended action based on risk level.
    
    Args:
        risk_level: Risk level (low/medium/high)
    
    Returns:
        Recommended action
    """
    action_map = {
        RISK_LOW: ACTION_CLEAR,
        RISK_MEDIUM: ACTION_REVIEW,
        RISK_HIGH: ACTION_PHYSICAL_INSPECTION,
    }
    
    return action_map.get(risk_level, ACTION_REVIEW)


def generate_reasoning(
    detected_items: List[str],
    declaration_comparison: Optional[Dict] = None,
    risk_level: str = RISK_LOW,
    has_occlusion: bool = False,
) -> str:
    """
    Generate human-readable reasoning for the inspection result.
    
    Args:
        detected_items: Items detected by VLM
        declaration_comparison: Declaration comparison results
        risk_level: Assessed risk level
        has_occlusion: Whether items are concealed
    
    Returns:
        Reasoning text
    """
    if not detected_items:
        return "No prohibited items detected. Baggage cleared."
    
    # Build reasoning
    parts = []
    
    # What was detected
    items_str = ", ".join(detected_items)
    parts.append(f"Detected items: {items_str}.")
    
    # Declaration comparison
    if declaration_comparison and declaration_comparison.get("has_declaration"):
        if declaration_comparison["undeclared_items"]:
            undeclared_str = ", ".join(declaration_comparison["undeclared_items"])
            parts.append(f"ALERT: Undeclared items found - {undeclared_str}.")
        elif declaration_comparison["declaration_match"]:
            parts.append("Declaration is consistent with scan.")
        else:
            parts.append("Declaration partially matches scan.")
    
    # Occlusion
    if has_occlusion:
        parts.append("Warning: Items appear to be intentionally concealed.")
    
    # Risk assessment
    if risk_level == RISK_HIGH:
        parts.append("High risk detected - immediate physical inspection required.")
    elif risk_level == RISK_MEDIUM:
        parts.append("Moderate risk - manual review recommended.")
    else:
        parts.append("Low risk assessment.")
    
    return " ".join(parts)


def process_vlm_response(
    vlm_answer: str,
    declared_items: Optional[List[str]] = None,
    metadata: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Complete post-processing pipeline for VLM response.
    
    Args:
        vlm_answer: Answer from VLM
        declared_items: Items declared by passenger
        metadata: Additional metadata (e.g., occlusion info)
    
    Returns:
        Processed result with risk assessment and recommendations
    """
    # Extract detected items from VLM answer
    detected_items = extract_items_from_text(vlm_answer)
    
    # Check for occlusion indicators in text
    has_occlusion = any(
        word in vlm_answer.lower()
        for word in ["conceal", "hidden", "occluded", "partially", "obscured"]
    )
    
    # Compare with declaration
    declaration_comparison = None
    if declared_items is not None:
        declaration_comparison = compare_with_declaration(detected_items, declared_items)
    
    # Assess risk
    risk_level = assess_risk_level(detected_items, declaration_comparison, has_occlusion)
    
    # Determine action
    recommended_action = determine_action(risk_level)
    
    # Generate reasoning
    reasoning = generate_reasoning(
        detected_items,
        declaration_comparison,
        risk_level,
        has_occlusion,
    )
    
    # Build result
    result = {
        "detected_items": detected_items,
        "risk_level": risk_level,
        "recommended_action": recommended_action,
        "reasoning": reasoning,
        "has_occlusion": has_occlusion,
    }
    
    # Add declaration comparison if available
    if declaration_comparison:
        result["declaration_comparison"] = declaration_comparison
    
    return result


# Example usage
if __name__ == "__main__":
    # Test case 1: Prohibited item detected, not declared
    vlm_answer_1 = "Detected items: a folding knife at center-left."
    declared_items_1 = ["clothing", "electronics"]
    
    result_1 = process_vlm_response(vlm_answer_1, declared_items_1)
    print("Test 1: Undeclared prohibited item")
    print(result_1)
    print()
    
    # Test case 2: Prohibited item detected, properly declared
    vlm_answer_2 = "Detected items: scissors in upper-right quadrant."
    declared_items_2 = ["scissors", "clothing"]
    
    result_2 = process_vlm_response(vlm_answer_2, declared_items_2)
    print("Test 2: Declared prohibited item")
    print(result_2)
    print()
    
    # Test case 3: Concealed item
    vlm_answer_3 = "Detected items: a utility knife at center, partially concealed."
    declared_items_3 = ["electronics"]
    
    result_3 = process_vlm_response(vlm_answer_3, declared_items_3)
    print("Test 3: Concealed undeclared item")
    print(result_3)
    print()
    
    # Test case 4: No prohibited items
    vlm_answer_4 = "No items detected in this scan."
    declared_items_4 = ["clothing"]
    
    result_4 = process_vlm_response(vlm_answer_4, declared_items_4)
    print("Test 4: Clean scan")
    print(result_4)
