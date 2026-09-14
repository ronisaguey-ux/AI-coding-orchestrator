import json
import logging
from typing import List, Dict, Any, Optional

FINDINGS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["finding_type", "description", "severity"],
        "properties": {
            "finding_type": {"type": "string"},
            "description": {"type": "string"},
            "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
            "location": {"type": "string"},
            "suggestion": {"type": "string"}
        },
        "additionalProperties": False
    }
}

MAX_FINDINGS = 100  # Maximum number of findings allowed per payload

class InvalidPayloadError(Exception):
    """Raised when a payload is invalid or malicious."""
    def __init__(self, message: str, validation_errors: Optional[List[str]] = None):
        super().__init__(message)
        self.validation_errors = validation_errors or []

def parse_findings_payload(payload: Any, strict: bool = True, error_collector: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Parse and validate a JSON payload against the findings schema.
    
    Args:
        payload: A JSON string or a dict/list already parsed.
        strict: If True, raise InvalidPayloadError for any validation failure.
               If False, return valid subset and collect errors for inspection.
        error_collector: If provided (and strict=False), validation error messages
                        are appended here so callers can programmatically inspect
                        them without relying on log output.
        
    Returns:
        A list of validated findings. In non-strict mode, invalid items are
        discarded and the valid subset is returned.
        
    Raises:
        InvalidPayloadError: If strict=True and payload is invalid or malicious.
    """
    validation_errors = []
    
    if isinstance(payload, str):
        payload = payload.strip()
        if not payload:
            error_msg = "Empty payload provided"
            if strict:
                raise InvalidPayloadError(error_msg, [error_msg])
            return []
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as e:
            error_msg = f"JSON decode error: {e}"
            validation_errors.append(error_msg)
            if strict:
                raise InvalidPayloadError(f"Invalid JSON payload: {e}", validation_errors)
            return []
    elif isinstance(payload, (dict, list)):
        data = payload
    else:
        error_msg = f"Unsupported payload type: {type(payload).__name__}"
        validation_errors.append(error_msg)
        if strict:
            raise InvalidPayloadError(error_msg, validation_errors)
        return []
    
    if data is None:
        error_msg = "Payload is None"
        validation_errors.append(error_msg)
        if strict:
            raise InvalidPayloadError(error_msg, validation_errors)
        return []
    
    # Normalize: if it's a dict with a "findings" key, extract it
    if isinstance(data, dict) and "findings" in data:
        data = data["findings"]
    
    # Ensure it's a list
    if not isinstance(data, list):
        error_msg = f"Payload must be a list, got {type(data).__name__}"
        validation_errors.append(error_msg)
        if strict:
            raise InvalidPayloadError(error_msg, validation_errors)
        return []
    
    # Enforce maximum findings limit to prevent memory exhaustion
    if len(data) > MAX_FINDINGS:
        error_msg = f"Payload exceeds maximum findings limit of {MAX_FINDINGS}"
        validation_errors.append(error_msg)
        if strict:
            raise InvalidPayloadError(error_msg, validation_errors)
        data = data[:MAX_FINDINGS]  # Truncate to first MAX_FINDINGS items
    
    validated = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            error_msg = f"Item at index {idx} is not a dict"
            validation_errors.append(error_msg)
            continue
        # Check required fields presence and type (strict)
        if not all(k in item for k in ("finding_type", "description", "severity")):
            error_msg = f"Item at index {idx} missing required fields"
            validation_errors.append(error_msg)
            continue
        # Validate required fields are non-empty strings with correct types
        required_valid = True
        for field in ("finding_type", "description", "severity"):
            val = item.get(field)
            if not isinstance(val, str):
                error_msg = f"Item at index {idx} field '{field}' must be a string, got {type(val).__name__}"
                validation_errors.append(error_msg)
                required_valid = False
                break
            if not val.strip():
                error_msg = f"Item at index {idx} has empty required field '{field}'"
                validation_errors.append(error_msg)
                required_valid = False
                break
        if not required_valid:
            continue
        if item["severity"] not in ("low", "medium", "high", "critical"):
            error_msg = f"Item at index {idx} has invalid severity: {item.get('severity')}"
            validation_errors.append(error_msg)
            continue
        # Validate optional fields types
        if "location" in item and not isinstance(item["location"], str):
            error_msg = f"Item at index {idx} has invalid location type"
            validation_errors.append(error_msg)
            continue
        if "suggestion" in item and not isinstance(item["suggestion"], str):
            error_msg = f"Item at index {idx} has invalid suggestion type"
            validation_errors.append(error_msg)
            continue
        # Reject additionalProperties
        allowed_keys = {"finding_type", "description", "severity", "location", "suggestion"}
        if any(k not in allowed_keys for k in item):
            error_msg = f"Item at index {idx} has unexpected properties"
            validation_errors.append(error_msg)
            continue
        # Build validated entry with allowed fields
        entry = {
            "finding_type": item["finding_type"],
            "description": item["description"],
            "severity": item["severity"]
        }
        if "location" in item:
            entry["location"] = item["location"]
        if "suggestion" in item:
            entry["suggestion"] = item["suggestion"]
        validated.append(entry)
    
    if validation_errors:
        if strict:
            raise InvalidPayloadError(
                f"Payload validation failed with {len(validation_errors)} error(s)",
                validation_errors
            )
        else:
            logging.warning(
                "parse_findings_payload: %d validation error(s) in non-strict mode: %s",
                len(validation_errors),
                "; ".join(validation_errors[-5:])
            )
            if error_collector is not None:
                error_collector.extend(validation_errors)

    return validated
