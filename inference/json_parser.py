"""
JSON parsing and validation utilities for structured VLM output.
Provides fallback parsing for cases where XGrammar guided generation is not used.
"""

import json
import re
from typing import Dict, Any, Optional, List


def parse_json_response(response: str | dict) -> Optional[Dict[str, Any]]:
    """
    Parse JSON from VLM response with fallback strategies.
    
    Args:
        response: Raw response string or dict from VLM
    
    Returns:
        Parsed JSON dict, or None if parsing failed
    """
    # If already a dict, return it
    if isinstance(response, dict):
        return response
    
    # Strategy 1: Direct JSON parsing
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract JSON from markdown code blocks
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(json_pattern, response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Find first { to last } (handles text before/after JSON)
    brace_start = response.find('{')
    brace_end = response.rfind('}')
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        try:
            json_str = response[brace_start:brace_end + 1]
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Strategy 4: Clean common formatting issues
    cleaned = response.strip()
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    return None


def validate_structured_output(data: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate structured output against expected schema.
    
    Args:
        data: Parsed JSON data
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required top-level keys
    required_keys = {"items", "total_count", "has_concealed_items"}
    missing = required_keys - set(data.keys())
    if missing:
        errors.append(f"Missing required keys: {missing}")
        return False, errors
    
    # Validate items
    if not isinstance(data["items"], list):
        errors.append("'items' must be a list")
        return False, errors
    
    valid_item_names = {
        "knife", "folding knife", "straight knife", "utility knife", "multi-tool knife",
        "scissors", "gun", "handgun", "pistol", "firearm", "explosive",
        "blade", "weapon", "prohibited item"
    }
    
    valid_locations = {
        "upper-left", "upper", "upper-right",
        "left", "center", "right",
        "lower-left", "lower", "lower-right",
        "center-left", "center-right", "upper-center", "lower-center"
    }
    
    for i, item in enumerate(data["items"]):
        # Check item structure
        if not isinstance(item, dict):
            errors.append(f"Item {i} is not a dict")
            continue
        
        # Check required item keys
        required_item_keys = {"name", "confidence", "location"}
        missing_item_keys = required_item_keys - set(item.keys())
        if missing_item_keys:
            errors.append(f"Item {i} missing keys: {missing_item_keys}")
        
        # Validate name
        if "name" in item and item["name"] not in valid_item_names:
            errors.append(f"Item {i} has invalid name: '{item['name']}' (should be one of {valid_item_names})")
        
        # Validate confidence
        if "confidence" in item:
            conf = item["confidence"]
            if not isinstance(conf, (int, float)):
                errors.append(f"Item {i} confidence must be numeric")
            elif not 0.0 <= conf <= 1.0:
                errors.append(f"Item {i} confidence {conf} not in [0.0, 1.0]")
        
        # Validate location
        if "location" in item and item["location"] not in valid_locations:
            errors.append(f"Item {i} has invalid location: '{item['location']}' (should be one of {valid_locations})")
    
    # Validate total_count
    if not isinstance(data["total_count"], int):
        errors.append("'total_count' must be an integer")
    elif data["total_count"] < 0:
        errors.append("'total_count' cannot be negative")
    elif data["total_count"] != len(data["items"]):
        errors.append(f"'total_count' ({data['total_count']}) doesn't match items length ({len(data['items'])})")
    
    # Validate has_concealed_items
    if not isinstance(data["has_concealed_items"], bool):
        errors.append("'has_concealed_items' must be boolean")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def extract_items_from_json(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract item details from validated JSON output.
    
    Args:
        data: Validated JSON output
    
    Returns:
        List of items with name, confidence, location
    """
    items = []
    for item in data.get("items", []):
        items.append({
            "name": item.get("name", "unknown"),
            "confidence": item.get("confidence", 0.0),
            "location": item.get("location", "center"),
        })
    return items


def fallback_to_text_extraction(response: str) -> List[Dict[str, Any]]:
    """
    Extract items from natural language response when JSON parsing fails.
    
    Args:
        response: Natural language response from VLM
    
    Returns:
        List of items with name, confidence (default), location (default)
    """
    items = []
    
    # Common item keywords
    item_keywords = [
        "knife", "knives", "gun", "guns", "pistol", "handgun", "firearm",
        "explosive", "blade", "scissors", "weapon", "prohibited"
    ]
    
    response_lower = response.lower()
    
    # Check for "no items" or similar
    if any(phrase in response_lower for phrase in ["no prohibited", "no items", "clean scan", "no threat"]):
        return []
    
    # Extract mentioned items
    for keyword in item_keywords:
        if keyword in response_lower:
            # Try to extract location context
            location = "center"
            location_patterns = {
                "upper-left": r"(upper[- ]left|top[- ]left)",
                "upper": r"(upper|top)\b(?![- ]left|[- ]right)",
                "upper-right": r"(upper[- ]right|top[- ]right)",
                "left": r"\bleft\b(?!top|upper|lower|bottom)",
                "right": r"\bright\b(?!top|upper|lower|bottom)",
                "lower-left": r"(lower[- ]left|bottom[- ]left)",
                "lower": r"(lower|bottom)\b(?![- ]left|[- ]right)",
                "lower-right": r"(lower[- ]right|bottom[- ]right)",
                "center": r"\b(center|middle)\b",
            }
            
            # Search for location near the keyword
            keyword_pos = response_lower.find(keyword)
            context = response_lower[max(0, keyword_pos - 50):min(len(response_lower), keyword_pos + 50)]
            
            for loc, pattern in location_patterns.items():
                if re.search(pattern, context):
                    location = loc
                    break
            
            # Normalize item name
            if keyword in ["knives"]:
                item_name = "knife"
            elif keyword in ["guns", "pistol", "handgun", "firearm"]:
                item_name = "gun"
            else:
                item_name = keyword
            
            items.append({
                "name": item_name,
                "confidence": 0.8,  # Default confidence for text extraction
                "location": location,
            })
    
    # Remove duplicates (keep first occurrence)
    seen_names = set()
    unique_items = []
    for item in items:
        if item["name"] not in seen_names:
            seen_names.add(item["name"])
            unique_items.append(item)
    
    return unique_items


def format_structured_output(items: List[Dict[str, Any]], has_concealed: bool = False) -> Dict[str, Any]:
    """
    Format items list into the standard structured output schema.
    
    Args:
        items: List of items with name, confidence, location
        has_concealed: Whether items are concealed
    
    Returns:
        Structured output dict
    """
    return {
        "items": items,
        "total_count": len(items),
        "has_concealed_items": has_concealed,
    }


# Example usage
if __name__ == "__main__":
    # Test JSON parsing
    test_cases = [
        # Case 1: Valid JSON
        '{"items": [{"name": "knife", "confidence": 0.95, "location": "center"}], "total_count": 1, "has_concealed_items": false}',
        
        # Case 2: JSON in markdown
        '```json\n{"items": [], "total_count": 0, "has_concealed_items": false}\n```',
        
        # Case 3: JSON with text
        'Based on analysis: {"items": [{"name": "gun", "confidence": 0.90, "location": "upper-left"}], "total_count": 1, "has_concealed_items": true}',
        
        # Case 4: Natural language (should fail JSON parsing)
        "I can see a knife in the center of the baggage and a gun in the upper-left area.",
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n=== Test Case {i} ===")
        print(f"Input: {test[:100]}...")
        
        result = parse_json_response(test)
        if result:
            print(f"✓ Parsed as JSON")
            is_valid, errors = validate_structured_output(result)
            if is_valid:
                print(f"✓ Valid schema")
                items = extract_items_from_json(result)
                print(f"  Items: {items}")
            else:
                print(f"✗ Invalid schema: {errors}")
        else:
            print(f"✗ JSON parsing failed, using text extraction")
            items = fallback_to_text_extraction(test)
            print(f"  Extracted items: {items}")
