import json

import yaml


def json_to_string(file_path: str) -> str:
    try:
        # Open the file in read mode
        with open(file_path, "r", encoding="utf-8") as file:
            # Parse the JSON data from the file
            data = json.load(file)

            # Print the data as a formatted string
            # indent=4 adds indentation for readability
            # ensure_ascii=False ensures special characters (like Greek letters) display correctly
            json_string = json.dumps(data, indent=4, ensure_ascii=False)
            # print(json_string)
            return json_string

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def json_to_pyObj(file_path: str) -> dict | list:
    try:
        # Open the file in read mode
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def load_yaml_sys_prompt(file_path: str, key: str = "system_prompt") -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
            return data[key]
    except Exception as e:
        raise Exception(f"Error loading YAML: {e}")
