import json
import pandas as pd
from intc import Parser
import hjson
from dlk.export import Export, ExportConfig

if __name__ == "__main__":
    # 1. Load Export Config
    config_dict = hjson.load(open("./config/export.jsonc", "r", encoding="utf-8"))
    export_config = ExportConfig._from_dict(config_dict["@export"])

    # 2. Prepare Dummy Data for Tracing
    dummy_data = [
        {"sentence": "A test sentence for ONNX tracing.", "uuid": "dummy-uuid-1"}
    ]
    input_df = pd.DataFrame(dummy_data)

    # 3. Export Model
    export_task = Export(config=export_config)
    export_task.export(input_df)
    print(f"Model exported to {export_config.output_path}")
