# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import io
import json
import logging
from typing import Any, Callable, Dict, List, Union
from uuid import uuid4

import hjson
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from intc import (
    MISSING,
    AnyField,
    Base,
    BoolField,
    DictField,
    FloatField,
    IntField,
    ListField,
    NestField,
    Parser,
    StrField,
    SubModule,
    asdict,
    cregister,
    init_config,
)
from PIL import Image

from dlk.display import Display
from dlk.online import OnlinePredict
from dlk.preprocess import PreProcessor
from dlk.utils.register import register, register_module_name

logger = logging.getLogger(__name__)


@st.cache_resource
def get_process(config: str) -> PreProcessor:
    """

    Args:
        config: the path to process config

    Returns:
        process instance

    """
    return PreProcessor(config, stage="online")


@st.cache_resource
def get_predict(config: str, checkpoint: str) -> OnlinePredict:
    """

    Args:
        config: the path to predict config
        checkpoint: the path to checkpoint

    Returns:
        predict instance

    """
    return OnlinePredict(config, checkpoint)


@st.cache_resource
def prepare_display(dump_display_config: str) -> Display:
    """load the display instance from the config

    Args:
        dump_display_config: the json dumps display config

    Returns:
        Display
    """

    configs = Parser(json.loads(dump_display_config)).parser_init()
    assert len(configs) == 1, f"You should not use '_search' for demo"
    display_config = configs[0]["@display"]
    display_name = register_module_name(display_config._module_name)
    display = register.get("display", display_name)(display_config)

    return display


if "predict" not in st.session_state:
    st.session_state.predict = None

if "process" not in st.session_state:
    st.session_state.process = None


class Demo(object):
    """Demo"""

    def __init__(
        self,
        display_config: Union[str, Dict],
        process_config: str = "/path/to/config",
        fit_config: str = "/path/to/config",
        checkpoint: str = "/path/to/checkpoint",
        scrolling=True,
    ):
        super(Demo, self).__init__()
        if isinstance(display_config, str):
            with open(display_config, "r") as f:
                display_config = hjson.load(f, object_pairs_hook=dict)
        self.display = prepare_display(json.dumps(display_config, sort_keys=True))
        self.scrolling = scrolling

        with st.sidebar:
            st.title("DLK Demo")
            process_config_value = st.text_input(
                "Process Config",
                value=None if process_config == "/path/to/config" else process_config,
                placeholder=process_config,
            )

            fit_config_value = st.text_input(
                "Main Config",
                value=None if fit_config == "/path/to/config" else fit_config,
                placeholder=fit_config,
            )
            checkpoint_path_value = st.text_input(
                "Checkpoint Path",
                value=None if checkpoint == "/path/to/checkpoint" else checkpoint,
                placeholder=checkpoint,
            )

            if st.button("Load Model"):
                self.load_model(
                    process_config_value, fit_config_value, checkpoint_path_value
                )

            if st.button("Clear Cache"):
                st.cache_resource.clear()

        st.header(self.display.config.title)
        if self.display.config.help:
            st.caption(self.display.config.help)
        st.markdown("--------")
        input_names = asdict(self.display.config.input)
        inputs = {}
        inputs["uuid"] = str(uuid4())
        inputs.update(asdict(self.display.config.hold))
        for input in input_names:
            if input_names[input] == "text":
                st.subheader(f"{input}:")
                inputs[input] = st.text_area(
                    label=" ",
                    placeholder=f"Please input the {input}",
                    value="",
                    height=3,
                )
            elif input_names[input] == "image":
                st.subheader(f"{input}:")
                image = st.file_uploader(label=" ", type=["png", "jpg", "jpeg"])
                if image:
                    image = Image.open(io.BytesIO(image.read()))
                    st.image(image, caption=f"Uploaded {input}", use_column_width=True)
                    inputs[input] = image
                else:
                    st.text("Please upload the right image")
            else:
                st.text("Please check the input type")
        if st.button("Submit"):
            self.generate_response(pd.DataFrame([inputs]))

    def load_model(self, process_config_value, fit_config_value, checkpoint_path_value):
        st.session_state.process = get_process(process_config_value)
        logger.info("Load Processor Done")
        st.session_state.predict = get_predict(fit_config_value, checkpoint_path_value)
        logger.info("Load Predictor Done")

    def generate_response(self, input_df):
        if (
            st.session_state.predict is not None
            and st.session_state.process is not None
        ):
            st.markdown("--------")
            st.subheader(f"{self.display.config.output_head}:")
            processed_data = st.session_state.process.fit(input_df)
            result = st.session_state.predict.predict(processed_data)
            if self.display.config.render_type == "html":
                components.html(
                    self.display.display(result[0]),
                    width=int(self.display.config.render_width * 1.1),
                    height=int(self.display.config.render_height * 1.1),
                    scrolling=self.scrolling,
                )
            elif self.display.config.render_type == "text":
                st.text_area(
                    "", self.display.display(result[0]), label_visibility="hidden"
                )
            elif self.display.config.render_type == "image":
                st.image(self.display.display(result[0]))
            else:
                st.text("Please check the render type")
            with st.expander("Json Result"):
                st.text(json.dumps(result[0], indent=4))
        else:
            st.text("Please load the model first")
