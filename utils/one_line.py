import yaml

class OneLineDict(dict):
    pass

def one_line_dict_representer(dumper, data):
    return dumper.represent_mapping(
        "tag:yaml.org,2002:map",
        data,
        flow_style=True
    )

yaml.SafeDumper.add_representer(OneLineDict, one_line_dict_representer)
