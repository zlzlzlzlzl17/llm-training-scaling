import os
from pathlib import Path, PurePosixPath

import modal

from llm_training_scaling.data_pipeline.common import MODAL_SHARED_PATH

SUNET_ID = os.environ.get("SUNET_ID")
if not SUNET_ID:
    raise RuntimeError("Set SUNET_ID before using the optional Modal data pipeline.")

(DATA_PATH := Path("data")).mkdir(exist_ok=True)

app = modal.App(f"data-{SUNET_ID}")
data_volume = modal.Volume.from_name(f"data-{SUNET_ID}", create_if_missing=True, version=2)
shared_data_volume = modal.Volume.from_name(
    "a4-shared-data", create_if_missing=True, version=2, environment_name="cs336-shared-data"
)


def build_image(*, include_tests: bool = False) -> modal.Image:
    image = modal.Image.debian_slim(python_version="3.12")
    image = image.uv_sync()
    image = image.add_local_dir(
        "src/llm_training_scaling",
        remote_path="/root/llm_training_scaling",
    )
    if include_tests:
        image = image.add_local_dir("tests/data_pipeline", remote_path="/root/tests/data_pipeline")
    return image


VOLUME_MOUNTS: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
    "/root/data": data_volume,
    str(MODAL_SHARED_PATH): shared_data_volume.read_only(),
}

MODAL_SECRETS = []
