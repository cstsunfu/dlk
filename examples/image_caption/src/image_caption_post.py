# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, List, Optional

import pandas as pd
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
    StrField,
    SubModule,
    cregister,
)

from dlk.data.postprocessor.generate_post import (
    TokenGeneratePostProcessor,
    TokenGeneratePostProcessorConfig,
)
from dlk.utils.register import register, register_module_name


@cregister("postprocessor", "image_caption")
class ImageCaptionPostProcessorConfig(TokenGeneratePostProcessorConfig):
    """"""

    class OriginInputMap:
        uuid = StrField(value="uuid", help="the uuid or the id of the sample")
        image = StrField(value="image", help="the image for caption")
        target = StrField(value="target", help="the target generation")
        image_url = StrField(value="", help="the url of the image, default is empty")

    origin_input_map = NestField(
        value=OriginInputMap,
        help="the origin input map of the processor, the key is the name of the processor needed key, the value is the provided data provided key",
    )


@register("postprocessor", "image_caption")
class ImageCaptionPostProcessor(TokenGeneratePostProcessor):
    """image caption postprocess"""

    def __init__(self, config: ImageCaptionPostProcessorConfig):
        super(ImageCaptionPostProcessor, self).__init__(config)
        self.config = config

    def _get_origin_data(self, one_origin: pd.Series) -> Dict:
        """

        Args:
            one_origin: the original data

        Returns:
            the gather origin data

        """
        origin = {}
        if self.config.origin_input_map.image_url:
            origin["image_url"] = one_origin[self.config.origin_input_map.image_url]
        origin["uuid"] = one_origin[self.config.origin_input_map.uuid]
        origin["target"] = one_origin[self.config.origin_input_map.target]
        return origin

    def do_predict(
        self,
        stage: str,
        list_batch_outputs: List[Dict],
        origin_data: pd.DataFrame,
        rt_config: Dict,
    ) -> List:
        """Process the model predict to human readable format

        Args:
            stage: train/test/etc.
            list_batch_outputs: a list of outputs
            origin_data: the origin pd.DataFrame data, there are some data not be able to convert to tensor
            rt_config:
                >>> current status
                >>> {
                >>>     "current_step": self.global_step,
                >>>     "current_epoch": self.current_epoch,
                >>>     "total_steps": self.num_training_steps,
                >>>     "total_epochs": self.num_training_epochs
                >>> }

        Returns:
            all predicts
        """
        results = []
        for outputs in list_batch_outputs:
            indexes = list(outputs[self.config.input_map.index])

            batch_generated = outputs[self.config.input_map.generated]
            for i, (index, generated) in enumerate(zip(indexes, batch_generated)):
                one_origin = origin_data.iloc[int(index)]
                one_ins = self._get_origin_data(one_origin)
                generate_result = []
                for i, (one_generate) in enumerate(generated):
                    generate_sent = self.tokenizer.decode(
                        list(one_generate["tokens"]),
                        skip_special_tokens=self.config.skip_special_tokens,
                    )
                    generate_result.append(
                        {
                            "generate": generate_sent,
                            "score": float(one_generate["score"]),
                        }
                    )
                    if self.config.return_all_generations == False:
                        break
                one_ins["generated"] = generate_result
                one_ins["predict_extend_return"] = self.gather_predict_extend_data(
                    outputs, i, self.config.predict_extend_return
                )
                if i == 0:
                    import json

                    one_ins.pop("image")

                    print("one_ins", json.dumps(one_ins, indent=4))
                results.append(one_ins)
        return results
