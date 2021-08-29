"""
The :mod:`mps.utils.configs` module defines `load_yaml_flag` function.
"""
import yaml


def load_yaml_flag(flag):
    yaml_flag = yaml.safe_load(flag)
    if not isinstance(yaml_flag, dict):
        raise ValueError('Got {}, expected YAML string.'.format(type(yaml_flag)))
    return yaml
