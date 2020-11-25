import multiprocessing
from pathlib import Path
from time import time
from zipfile import ZipFile
import logging
from typing import Union
import os

DATA = Path(r"D:\deepfakes\data\train")
logging.basicConfig(filename="extract2.log", level=logging.INFO)
zipfiles = sorted(list(DATA.glob("**/*.zip")), key=lambda x: x.stem)


def extract_zip(zipfile: Union[str, Path]) -> None:
    with ZipFile(zipfile) as zip_file:
        for file in zip_file.namelist():
            zip_file.extract(member=file, path=zipfile.parent)
    logging.info(f'Finished extracting {zipfile.stem}')
    os.remove(zipfile)


if __name__ == "__main__":
    start = time()
    with multiprocessing.Pool() as pool:
        pool.map(extract_zip, zipfiles)

    logging.info(f"Extracted all zip files in {time() - start} seconds!")