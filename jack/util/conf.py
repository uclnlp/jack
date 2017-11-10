import yaml


def load_config(conf_yaml_file: str):
    """
    Loads a YAML configuration file. If the file contains a `parent_config` field,
    the parent config is (recursively) added to the return dictionary.
    Args:
        conf_yaml_file: the name of the yaml config file to load

    Returns: a dictionary with config keys to values
    """
    with open(conf_yaml_file, "r") as file:
        loaded = yaml.load(file)
        if "parent_config" in loaded:
            parent_conf = load_config(loaded['parent_config'])
            return {
                **parent_conf,
                **loaded
            }
        else:
            return loaded
