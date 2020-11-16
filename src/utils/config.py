import os
import configparser as cp
import ast
import types


def get_config(custom_ini_file):

    file_conf = cp.ConfigParser()
    root = os.path.split(custom_ini_file)[0]

    # Read in configs in hierarchy
    file_conf.read(os.path.join(root, 'default.ini'))
    file_conf.read(custom_ini_file)

    conf = {}

    # Convert dictionaries to namespaces
    for section_name in file_conf.sections():
        d = {}
        for (key, val) in file_conf.items(section_name):
            d[key] = ast.literal_eval(val)

        item = types.SimpleNamespace(**d)
        conf[section_name] = item
    x = types.SimpleNamespace(**conf)

    return x