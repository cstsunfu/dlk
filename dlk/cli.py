# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import ast
import json
import logging
import os
import sys
from typing import Any, Dict, List

import hjson
import pandas as pd
from ray import serve

from dlk.export import Export, ExportConfig
from dlk.preprocess import PreProcessor
from dlk.ray_server import RayServeGlobalConfig, build_app
from dlk.train import Train
from dlk.utils.logger import setup_logger

logger = logging.getLogger(__name__)


def parse_overrides(overrides: List[str]) -> Dict[str, Any]:
    """Parses a list of key=value strings into a nested dictionary.

    This is used to dynamically override configurations at runtime.
    Example:
        Input: ["@fit.log_dir='./new_logs'", "@fit.@trainer.max_epochs=5"]
        Output: {"@fit": {"log_dir": "./new_logs", "@trainer": {"max_epochs": 5}}}

    Args:
        overrides: A list of string assignments.

    Returns:
        A nested dictionary representing the configuration updates.
    """
    update_dict = {}
    for item in overrides:
        if "=" not in item:
            logger.warning(f"Ignoring invalid override (missing '='): {item}")
            continue

        k_str, v_str = item.split("=", 1)
        keys = k_str.split(".")

        # Auto type inference (int, float, bool, list, etc.)
        try:
            val = ast.literal_eval(v_str)
        except (ValueError, SyntaxError):
            val = v_str  # Keep as string if it cannot be evaluated

        # Build nested dictionary
        d = update_dict
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = val

    return update_dict


def run_process(args: argparse.Namespace, overrides: Dict[str, Any]):
    """Executes the data preprocessing pipeline.

    Args:
        args: Parsed command-line arguments.
        overrides: Dictionary of config overrides.
    """
    processor = PreProcessor(args.config, stage="train", update_config=overrides)

    data_dict = {}
    if args.train_data:
        data_dict["train"] = args.train_data
    if args.valid_data:
        data_dict["valid"] = args.valid_data
    if args.test_data:
        data_dict["test"] = args.test_data

    logger.info("Starting offline data preprocessing...")
    processor.fit(data_dict)
    logger.info("Preprocessing completed successfully.")


def run_train(args: argparse.Namespace, overrides: Dict[str, Any]):
    """Executes the model training pipeline.

    Args:
        args: Parsed command-line arguments.
        overrides: Dictionary of config overrides.
    """
    trainer = Train(
        config=args.config,
        checkpoint=args.checkpoint,
        strict=args.strict,
        update_config=overrides,
    )
    logger.info("Starting model training...")
    trainer.run()
    logger.info("Training completed successfully.")


def run_export(args: argparse.Namespace, overrides: Dict[str, Any]):
    """Executes the ONNX export pipeline.

    Args:
        args: Parsed command-line arguments.
        overrides: Dictionary of config overrides.
    """
    config_dict = hjson.load(open(args.config, "r", encoding="utf-8"))

    # Merge overrides into the loaded config manually for export
    # (Since ExportConfig doesn't natively take update_config in its current API,
    # we merge it before instantiating the Base model)
    from intc.utils import update_dict

    if overrides and "@export" in overrides:
        update_dict(config_dict["@export"], overrides["@export"])

    export_config = ExportConfig._from_dict(config_dict["@export"])
    export_task = Export(config=export_config)

    # Load dummy data for ONNX tracing
    if not args.dummy_data or not os.path.exists(args.dummy_data):
        raise FileNotFoundError(
            f"Dummy data file for tracing not found: {args.dummy_data}"
        )

    with open(args.dummy_data, "r", encoding="utf-8") as f:
        dummy_list = json.load(f)
    input_df = pd.DataFrame(dummy_list)

    logger.info("Starting ONNX export...")
    export_task.export(input_df)


def run_serve(args: argparse.Namespace, overrides: Dict[str, Any]):
    """Starts the Ray Serve multi-model deployment.

    Args:
        args: Parsed command-line arguments.
        overrides: Dictionary of config overrides.
    """
    import ray

    # Initialize Ray cluster
    ray.init(ignore_reinit_error=True)

    # Currently Ray build_app expects a file path. To support overrides,
    # we can temporarily dump the merged config to a file.
    config_dict = hjson.load(open(args.config, "r", encoding="utf-8"))
    if overrides:
        from intc.utils import update_dict

        update_dict(config_dict, overrides)

    temp_config_path = args.config + ".temp.jsonc"
    with open(temp_config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f)

    try:
        app = build_app(temp_config_path)
        logger.info(f"Deploying Ray Serve app on {args.host}:{args.port}...")
        serve.run(app, host=args.host, port=args.port)

        logger.info("Ray Serve is running. Press Ctrl+C to stop.")
        import time

        while True:
            time.sleep(1)
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


def main():
    """Main entry point for the DLK CLI."""
    setup_logger()

    parser = argparse.ArgumentParser(
        description="DLK (Deep Learning Kit) Command Line Interface"
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Subcommands"
    )

    # --- Process Command ---
    parser_process = subparsers.add_parser(
        "process", help="Run offline data preprocessing"
    )
    parser_process.add_argument(
        "-c", "--config", required=True, help="Path to processor.jsonc"
    )
    parser_process.add_argument(
        "--train_data", type=str, default="", help="Path to train data source"
    )
    parser_process.add_argument(
        "--valid_data", type=str, default="", help="Path to valid data source"
    )
    parser_process.add_argument(
        "--test_data", type=str, default="", help="Path to test data source"
    )

    # --- Train Command ---
    parser_train = subparsers.add_parser("train", help="Train a deep learning model")
    parser_train.add_argument("-c", "--config", required=True, help="Path to fit.jsonc")
    parser_train.add_argument(
        "-ckpt",
        "--checkpoint",
        type=str,
        default="",
        help="Path to checkpoint to resume/finetune from",
    )
    parser_train.add_argument(
        "--strict",
        action="store_true",
        help="Strictly enforce state_dict matching when loading checkpoint",
    )

    # --- Export Command ---
    parser_export = subparsers.add_parser(
        "export", help="Export a trained model to ONNX"
    )
    parser_export.add_argument(
        "-c", "--config", required=True, help="Path to export.jsonc"
    )
    parser_export.add_argument(
        "-d",
        "--dummy_data",
        required=True,
        help="Path to a JSON file containing dummy data for ONNX tracing",
    )

    # --- Serve Command ---
    parser_serve = subparsers.add_parser("serve", help="Deploy models using Ray Serve")
    parser_serve.add_argument(
        "-c", "--config", required=True, help="Path to serve.jsonc"
    )
    parser_serve.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind the API Gateway",
    )
    parser_serve.add_argument(
        "-p", "--port", type=int, default=8000, help="Port to bind the API Gateway"
    )

    # Parse known args. Remaining args are treated as config overrides.
    args, unknown_args = parser.parse_known_args()
    overrides = parse_overrides(unknown_args)

    if overrides:
        logger.info(
            f"Applied Configuration Overrides: {json.dumps(overrides, indent=2)}"
        )

    # Route to the appropriate function
    if args.command == "process":
        run_process(args, overrides)
    elif args.command == "train":
        run_train(args, overrides)
    elif args.command == "export":
        run_export(args, overrides)
    elif args.command == "serve":
        run_serve(args, overrides)


if __name__ == "__main__":
    main()
