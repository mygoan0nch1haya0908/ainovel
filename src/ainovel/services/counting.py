def count_visible_characters(body: str) -> int:
    return sum(1 for character in body if not character.isspace())
