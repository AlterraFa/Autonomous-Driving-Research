import configparser

def get_nested_config(path):
    config = configparser.ConfigParser()
    config.read(path)
    
    nested_dict = {}
    for section in config.sections():
        parts = section.split('.')
        d = nested_dict
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = dict(config.items(section))
    return nested_dict