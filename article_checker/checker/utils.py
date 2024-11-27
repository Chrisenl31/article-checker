def check_abstract_structure(abstract):
    required_sections = ["background", "purpose", "method", "results", "conclusion"]
    abstract_lower = abstract.lower()
    for section in required_sections:
        if section not in abstract_lower:
            return False, f"Missing section: {section.capitalize()}"
    return True, "Abstract structure is correct"
