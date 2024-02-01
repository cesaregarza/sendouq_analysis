import logging

import requests


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


def setup_logging() -> None:
    """Sets up default logging settings for the project"""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


def get_droplet_id(do_api_token: str, droplet_name: str) -> str:
    """Gets the id of a droplet from its name

    Args:
        do_api_token (str): The DigitalOcean API token
        droplet_name (str): The name of the droplet

    Returns:
        str: The id of the droplet
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {do_api_token}",
    }
    response = requests.get(
        "https://api.digitalocean.com/v2/droplets",
        headers=headers,
        params={"per_page": 200},
    )
    response.raise_for_status()
    droplets = response.json().get("droplets", [])
    for droplet in droplets:
        if droplet["name"] == droplet_name:
            return droplet["id"]
    raise ValueError(f"No droplet with name {droplet_name} found")


def delete_droplet(do_api_token: str, droplet_id: str) -> None:
    """Deletes a droplet

    Args:
        do_api_token (str): The DigitalOcean API token
        droplet_id (str): The id of the droplet
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {do_api_token}",
    }
    response = requests.delete(
        f"https://api.digitalocean.com/v2/droplets/{droplet_id}",
        headers=headers,
    )
    response.raise_for_status()
