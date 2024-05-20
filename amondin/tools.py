"""
Module containing useful tools for amondin
"""

import yaml


def get_secret(path2yaml: str, key: str):
    """
    Function to retrieve a value from a secrets.yaml
    :param path2yaml:
    :param key:
    :return:
    """
    with open(path2yaml, "r", encoding="utf-8") as file:
        secrets = yaml.safe_load(file)

    return secrets[key]
