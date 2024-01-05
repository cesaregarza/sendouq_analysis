def camel_to_snake(camel: str) -> str:
    """Converts a camelCase string to snake_case

    Args:
        camel (str): The camelCase string to convert

    Returns:
        str: The snake_case string
    """
    snake = ""
    for char in camel:
        if char.isupper():
            snake += "_"
        snake += char.lower()
    return snake
