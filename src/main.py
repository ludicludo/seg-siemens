import argparse
import logging
import numpy as np
from pathlib import Path
from seg_siemens.segmentation import (
    read_dicom_segmentation,
    create_seg_object,
    load_dicom_series,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seg", help="input segmentation dicom file")
    parser.add_argument("-o", "--out", help="output segmentation dicom file")
    return parser.parse_args()


def main():
    
    args = argument()
    seg_file_path = args.seg
    logger.debug(seg_file_path)
    # Lecture du fichier DICOM de segmentation
    segment_file = Path(seg_file_path)
    mask = read_dicom_segmentation(segment_file)

    print(f"Shape: {mask.shape}")
    print(f"Dtype: {mask.dtype}")
    print(f"Unique values: {np.unique(mask)}")
    print(f"volume cm3: {mask.sum()}")

    series_number = 100
    pt_datasets = load_dicom_series(segment_file.parent, "PT")
    seg_object = create_seg_object(pt_datasets, mask, series_number)
    if args.out:
        seg_object.save_as(segment_file.parent / args.out)
    else:
        seg_object.save_as((segment_file.parent / (segment_file.name + "_new")).with_suffix(".dcm"))



if __name__ == "__main__":
    main()
