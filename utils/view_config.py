import jsoneditor
import hjson
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="The config path.",
    )


    args = parser.parse_args()
    if os.path.isdir(args.config):
        config_path = os.path.join(args.config, 'config.json')
    else:
        config_path = args.config
    assert os.path.exists(config_path)

    config = hjson.load(open(config_path), object_pairs_hook=dict)

    jsoneditor.editjson(config)
