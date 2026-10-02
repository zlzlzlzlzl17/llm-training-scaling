import gzip
import shutil
import tempfile
import urllib.request
from functools import cached_property
from io import BytesIO
from pathlib import Path

import modal
import polars as pl
from furu import Furu
from warcio.archiveiterator import ArchiveIterator
from warcio.warcwriter import WARCWriter

from llm_training_scaling.data_pipeline.common import get_shared_assets_path
from llm_training_scaling.data_pipeline.language import identify_language
from llm_training_scaling.data_pipeline.modal_utils import VOLUME_MOUNTS, app, build_image
import socket
import time
import urllib.error


BASE_URL = "https://data.commoncrawl.org/"



def _download_with_retries(
    url: str,
    output_path: str | Path,
    *,
    retries: int = 8,
    timeout: int = 120,
) -> None:
    """Download a file with timeout, retries, and atomic replacement."""
    output_path = Path(output_path)
    partial_path = output_path.with_name(output_path.name + ".part")

    for attempt in range(1, retries + 1):
        partial_path.unlink(missing_ok=True)

        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 "
                        "(compatible; CS336 data filtering pipeline)"
                    )
                },
            )

            downloaded_bytes = 0

            with urllib.request.urlopen(
                request,
                timeout=timeout,
            ) as response:
                content_length = response.headers.get("Content-Length")
                expected_bytes = (
                    int(content_length)
                    if content_length is not None
                    else None
                )

                with partial_path.open("wb") as output_file:
                    while True:
                        chunk = response.read(1024 * 1024)

                        if not chunk:
                            break

                        output_file.write(chunk)
                        downloaded_bytes += len(chunk)

            if (
                expected_bytes is not None
                and downloaded_bytes != expected_bytes
            ):
                raise OSError(
                    "Incomplete download: "
                    f"expected {expected_bytes} bytes, "
                    f"received {downloaded_bytes} bytes"
                )

            if downloaded_bytes == 0:
                raise OSError("Downloaded file is empty")

            partial_path.replace(output_path)
            return

        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            socket.timeout,
            ConnectionError,
            OSError,
        ) as error:
            partial_path.unlink(missing_ok=True)

            if attempt >= retries:
                raise RuntimeError(
                    f"Failed to download {url} after "
                    f"{retries} attempts"
                ) from error

            wait_seconds = min(60, 2 ** attempt)

            print(
                f"Download failed ({attempt}/{retries}): "
                f"{type(error).__name__}: {error}; "
                f"retrying in {wait_seconds}s",
                flush=True,
            )

            time.sleep(wait_seconds)

class _EnglishWetFile(Furu[Path]):
    chunk_urls: tuple[str, ...]

    def _create(self) -> Path:
        output_path = self.data_dir / "data.warc.wet.gz"

        self.logger.info("Loading English language identifier")

        def is_english(text: str) -> bool:
            """Return whether text is confidently identified as English."""
            language, confidence = identify_language(text)
            return language == "en" and confidence >= 0.7

        total_text = 0
        skipped_text = 0

        self.logger.info(
            "Processing WET chunk (%d files)",
            len(self.chunk_urls),
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            dir="/tmp",
            suffix=f".{output_path.name}",
        ) as temp_output_file:
            temp_output_path = Path(temp_output_file.name)

        with gzip.open(temp_output_path, "wb") as output_stream:
            writer = WARCWriter(output_stream, gzip=False)

            for wet_url in self.chunk_urls:
                local_wet_path = Path("/tmp") / wet_url.split("/")[-1]

                if not local_wet_path.exists():
                    self.logger.info(
                        "Downloading %s to %s",
                        wet_url,
                        local_wet_path,
                    )
                    _download_with_retries(wet_url, local_wet_path)
                else:
                    self.logger.info(
                        "Using cached WET file %s",
                        local_wet_path,
                    )

                with gzip.open(local_wet_path, "rb") as input_stream:
                    for record in ArchiveIterator(input_stream):
                        # Preserve non-document WARC records unchanged.
                        if record.rec_type != "conversion":
                            writer.write_record(record)
                            continue

                        payload = record.content_stream().read()
                        text = payload.decode(
                            "utf-8",
                            errors="replace",
                        )

                        total_text += len(text)

                        if is_english(text):
                            # content_stream().read() consumed the record,
                            # so restore its raw stream before writing.
                            record.raw_stream = BytesIO(payload)
                            writer.write_record(record)
                        else:
                            skipped_text += len(text)

        shutil.copy2(temp_output_path, output_path)
        temp_output_path.unlink(missing_ok=True)

        kept_percentage = (
            100 * (total_text - skipped_text) / total_text
            if total_text
            else 0
        )

        self.logger.info(
            "Finished WET chunk: wrote %s, kept %.2f%% of text",
            output_path,
            kept_percentage,
        )

        return output_path

    @cached_property
    def storage_root(self) -> Path:
        return get_shared_assets_path() / "furu"


@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    timeout=60 * 60 * 12,
    max_containers=128,
)
def make_wet_file_on_modal(
    wet_file: _EnglishWetFile,
) -> Path:
    return wet_file.load_or_create()


class EnglishWetFiles(Furu[list[Path]]):
    # Local learning configuration:
    # download only one raw Common Crawl WET file.
    n_files: int = 1
    group_size: int = 1

    shuffle_seed: int = 336
    crawl_id: str = "CC-MAIN-2026-17"

    def _create(self) -> list[Path]:
        if self.n_files <= 0:
            raise ValueError("n_files must be positive")

        if self.group_size <= 0:
            raise ValueError("group_size must be positive")

        if self.n_files % self.group_size != 0:
            raise ValueError(
                "n_files must be divisible by group_size"
            )

        wet_paths_url = (
            f"{BASE_URL}"
            f"crawl-data/{self.crawl_id}/wet.paths.gz"
        )

        self.logger.info(
            "Loading WET paths from %s",
            wet_paths_url,
        )

        wet_path_series = (
            pl.read_csv(
                wet_paths_url,
                has_header=False,
                new_columns=["wet_path"],
            )
            .sample(
                n=self.n_files,
                shuffle=True,
                seed=self.shuffle_seed,
                with_replacement=False,
            )["wet_path"]
        )

        wet_urls = [
            BASE_URL + wet_path
            for wet_path in wet_path_series.to_list()
        ]

        self.logger.info(
            "Selected %d WET files for crawl %s",
            len(wet_urls),
            self.crawl_id,
        )

        wet_files: list[_EnglishWetFile] = []

        for chunk_idx in range(
            0,
            len(wet_urls),
            self.group_size,
        ):
            chunk_urls = tuple(
                wet_urls[
                    chunk_idx : chunk_idx + self.group_size
                ]
            )

            wet_files.append(
                _EnglishWetFile(
                    chunk_urls=chunk_urls,
                )
            )

        self.logger.info(
            "Making %d English WET files",
            len(wet_files),
        )

        wet_data_paths: list[Path] = []

        if modal.is_local():
            self.logger.info(
                "Downloading WET files locally"
            )

            for wet_file_idx, wet_file in enumerate(
                wet_files,
                start=1,
            ):
                wet_data_paths.append(
                    wet_file.load_or_create()
                )

                self.logger.info(
                    "Completed %d/%d WET chunks",
                    wet_file_idx,
                    len(wet_files),
                )
        else:
            self.logger.info(
                "Downloading WET files remotely"
            )

            wet_data_paths = list(
                make_wet_file_on_modal.map(wet_files)
            )

            self.logger.info(
                "Completed %d remote WET chunks",
                len(wet_data_paths),
            )

            repo_path = (
                get_shared_assets_path()
                / "english-wet-data"
            )

            repo_path.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.logger.info(
                "Linking remote WET outputs into %s",
                repo_path,
            )

            source_link = repo_path / ".source"

            if (
                source_link.exists()
                or source_link.is_symlink()
            ):
                self.logger.info(
                    "Replacing existing source link %s",
                    source_link,
                )
                source_link.unlink()

            source_link.symlink_to(self.data_dir)

            self.logger.info(
                "Linked source data directory %s -> %s",
                source_link,
                self.data_dir,
            )

            for wet_data_idx, wet_data_path in enumerate(
                wet_data_paths
            ):
                link_path = (
                    repo_path
                    / f"{wet_data_idx:05d}-"
                    f"{wet_data_path.name}"
                )

                if (
                    link_path.exists()
                    or link_path.is_symlink()
                ):
                    self.logger.info(
                        "Replacing existing WET chunk link %s",
                        link_path,
                    )
                    link_path.unlink()

                link_path.symlink_to(wet_data_path)

                self.logger.info(
                    "Linked WET chunk %d: %s -> %s",
                    wet_data_idx,
                    link_path,
                    wet_data_path,
                )

        self.logger.info(
            "Finished creating %d English WET files",
            len(wet_data_paths),
        )

        return wet_data_paths

    @cached_property
    def storage_root(self) -> Path:
        return get_shared_assets_path() / "furu"
