from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pydicom
from pydicom.misc import is_dicom
from pydicom import Dataset
from pydicom.sr.codedict import codes
from highdicom.seg import (
    SegmentDescription,
    SegmentAlgorithmTypeValues,
)
from highdicom import AlgorithmIdentificationSequence
from highdicom.seg import Segmentation
from highdicom.seg import SegmentationTypeValues
from highdicom.uid import UID

import logging

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def validate_dicom_file(file_path: Path) -> bool:
    """Validate the input DICOM file path."""
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    if not file_path.is_file():
        raise ValueError(f"The path {file_path} is not a file.")
    if not is_dicom(file_path):
        raise ValueError(f"The file {file_path} is not a DICOM file.")
    return True


def validate_dicom_data(dicom_data: pydicom.Dataset) -> bool:
    """Validate the DICOM data."""
    if not hasattr(dicom_data, "pixel_array"):
        raise ValueError("The DICOM data does not contain pixel array.")
    if dicom_data.pixel_array.ndim not in (2, 3):
        raise ValueError("The DICOM pixel array should be 2 or 3-dimensional.")
    return True


def read_dicom_segmentation(file_path: Path) -> np.ndarray:
    """
    Read a DICOM segmentation file and return a binary mask.

    Args:
        file_path: Path to the DICOM segmentation file.

    Returns:
        Binary mask as a NumPy array with dtype uint8.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a DICOM file or the DICOM data is invalid.
    """
    validate_dicom_file(file_path)

    dicom_data = pydicom.dcmread(file_path)
    validate_dicom_data(dicom_data)

    pixel_array = dicom_data.pixel_array

    # Convert the pixel array to a binary mask
    binary_mask = (pixel_array > 0).astype(np.uint8)

    return binary_mask


def load_dicom_series(
    directory: Path, expected_modality: Optional[str] = None
) -> List[Dataset]:
    """
    Charge et traite les fichiers DICOM selon leur modalité.

    Args:
        directory (Path): Chemin vers le dossier contenant les fichiers DICOM.
        expected_modality (Optional[str], optional): Modalité attendue. Si None, aucune vérification.

    Returns:
        List[Dataset]: Liste des objets DICOM, triée si modalité PET/CT, non triée sinon.
    """
    dicom_files = []
    for f in directory.iterdir():
        if pydicom.misc.is_dicom(str(f)):
            try:
                dicom_file = pydicom.dcmread(str(f), force=True)
                if (
                    expected_modality
                    and hasattr(dicom_file, "Modality")
                    and dicom_file.Modality != expected_modality
                ):
                    logging.warning(
                        f"Modalité incorrecte pour {f}: {dicom_file.Modality}"
                    )
                    continue
                dicom_files.append(dicom_file)
            except Exception as e:
                logging.error(f"Erreur lecture {f}: {str(e)}")

    if (
        dicom_files
        and hasattr(dicom_files[0], "Modality")
        and dicom_files[0].Modality in ["PT", "CT"]
    ):
        dicom_files.sort(
            key=lambda x: (
                float(x.ImagePositionPatient[2])
                if hasattr(x, "ImagePositionPatient") and x.ImagePositionPatient
                else 0
            )
        )
    return dicom_files

def get_num_frames_from_segmentation(seg_dataset: Dataset) -> int:
    """Extraire le nombre de frames du dataset de segmentation"""
    if not hasattr(seg_dataset, "PerFrameFunctionalGroupSequence"):
        raise ValueError(
            "La segmentation n'a pas d'attribut PerFrameFunctionalGroupSequence."
        )
    
    num_frames = len(seg_dataset.PerFrameFunctionalGroupSequence)
    logger.info(f"Nombre de frames dans la segmentation : {num_frames}")
    return num_frames


def get_num_segments_from_segmentation(seg_dataset: Dataset) -> int:
    """Extraire le nombre de segments du dataset de segmentation"""
    if not hasattr(seg_dataset, "SegmentSequence"):
        raise ValueError("La segmentation n'a pas d'attribut SegmentSequence.")
    
    num_segments = len(seg_dataset.SegmentSequence)
    logger.info(f"Nombre de segments : {num_segments}")
    return num_segments

def prepare_mask_array_for_seg(
    mask_array_input: np.ndarray,
    num_frames: int,
    num_segments: int = 1,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Préparer le masque pour la création d'une segmentation DICOM SEG.
    
    Cette fonction gère TROIS cas d'entrée majeurs :
    
    1. Masque 2D : (height, width) 
       → Pour 1 frame, 1 segment
       → Sortie : (1, height, width, 1)
    
    2. Masque 3D SIMPLE : (num_frames, height, width)
       → Quand on a UN seul segment
       → Sortie : (num_frames, height, width, 1)
    
    3. Masque 3D CONCATÉNÉ : (num_segments * num_frames, height, width)
       → Quand les segments sont EMPILÉS linéairement
       → Sortie : (num_frames, height, width, num_segments)
    
    4. Masque 4D : (num_frames, height, width, num_segments)
       → Déjà au bon format
       → Sortie : inchangé
    
    Args:
        mask_array_input: Masque d'entrée en 2D, 3D ou 4D
        num_frames: Nombre de frames (REQUIS pour détecter 3D concaténé)
        num_segments: Nombre de segments (default 1)
    
    Returns:
        Tuple[np.ndarray, Tuple]: (mask_array_reshaped, final_shape)
    
    Raises:
        ValueError: Si les dimensions ne correspondent pas
    """
    
    logger.info("Shape d'entrée du masque : %s", 
                mask_array_input.shape)
    logger.info("Paramètres : num_frames= %d, num_segments= %d",
                 num_frames, num_segments)
    
    # Cas 1 : Masque 2D (height, width)
    if mask_array_input.ndim == 2:
        height, width = mask_array_input.shape
        
        if num_frames != 1:
            raise ValueError(
                f"Masque 2D reçu mais num_frames={num_frames} (attendu 1). "
                f"Utilisez un masque 3D pour plusieurs frames."
            )
        
        mask_reshaped = mask_array_input.reshape((1, height, width, 1))
        logger.info("Cas 1 (2D) : reshape en 4D :", mask_reshaped.shape)
        return mask_reshaped, (1, height, width, 1)
    
    # Cas 2 & 3 : Masque 3D - Détecter si simple ou concaténé
    elif mask_array_input.ndim == 3:
        dim0, height, width = mask_array_input.shape
        
        # CAS 3 : MASQUE CONCATÉNÉ
        # Quand dim0 = num_segments * num_frames
        if dim0 == num_segments * num_frames and num_segments > 1:
            logger.info(
                "Cas 3 (3D concaténé détecté) : %d = %d × %d",
                dim0, num_segments, num_frames
            )
            
            # Découper en segments
            mask_4d = np.zeros(
                (num_frames, height, width, num_segments),
                dtype=mask_array_input.dtype
            )
            
            # Chaque segment est empilé linéairement
            for seg_idx in range(num_segments):
                start_frame = seg_idx * num_frames
                end_frame = (seg_idx + 1) * num_frames
                mask_4d[:, :, :, seg_idx] = mask_array_input[start_frame:end_frame, :, :]
                logger.info("  Segment %s : frames %s-{end_frame-1}", 
                            seg_idx, start_frame)
            
            logger.info("Résultat :", mask_4d.shape)
            return mask_4d, (num_frames, height, width, num_segments)
        
        # CAS 2 : MASQUE 3D SIMPLE
        # Quand dim0 = num_frames (donc num_segments = 1)
        elif dim0 == num_frames and num_segments == 1:
            logger.info("Cas 2 (3D simple) : reshape en 4D")
            
            mask_reshaped = mask_array_input.reshape((num_frames, height, width, 1))
            logger.info("Résultat :", mask_reshaped.shape)
            return mask_reshaped, (num_frames, height, width, 1)
        
        else:
            # Ambiguïté : on ne sait pas comment interpréter
            raise ValueError(
                f"Ambiguïté dans les dimensions du masque 3D : {mask_array_input.shape}\n"
                f"  - Si c'est UN segment : dim0 ({dim0}) devrait = num_frames ({num_frames})\n"
                f"  - Si c'est MULTI-segments : dim0 ({dim0}) devrait = num_frames ({num_frames}) × num_segments ({num_segments}) = {num_frames * num_segments}\n"
                f"Passez un masque 4D (frames, height, width, segments) pour éviter l'ambiguïté."
            )
    
    # Cas 4 : Masque 4D (frames, height, width, segments)
    elif mask_array_input.ndim == 4:
        frames, height, width, segments = mask_array_input.shape
        
        if frames != num_frames:
            raise ValueError(
                f"Masque 4D : frames ({frames}) != num_frames ({num_frames})"
            )
        
        if segments != num_segments:
            raise ValueError(
                f"Masque 4D : segments ({segments}) != num_segments ({num_segments})"
            )
        
        logger.info("Cas 4 (4D) : format déjà correct")
        return mask_array_input, (frames, height, width, segments)
    
    else:
        raise ValueError(
            f"Dimensions invalides : {mask_array_input.ndim}D. "
            f"Attendu 2D, 3D ou 4D"
        )


def check_series_compatibility(
    dicom_series: List[Dataset], 
    mask_array: np.ndarray,
    num_segments: int = 1
) -> bool:
    """
    Vérifie si une série DICOM est compatible avec un masque binaire.

    Args:
        dicom_series (list): Liste des objets DICOM.
        mask_array (numpy.ndarray): Masque binaire.

    Returns:
        bool: True si la série DICOM est compatible avec le masque binaire, False sinon.
    """
    """
    Vérifie si une série DICOM est compatible avec un masque binaire.
    """
    # Vérifier que mask_array a 4 dimensions
    if mask_array.ndim != 4:
        logger.error(
            "mask_array devrait avoir 4 dimensions (frames, height, width, segments), "
            "mais a %s dimensions. Shape: %s",
            mask_array.ndim, mask_array.shape
        )
        return False
    
    num_frames, height, width, segments = mask_array.shape
    
    # Vérifier le nombre de segments
    if segments != num_segments:
        logger.error(
            "Nombre de segments incompatible : mask a %s segments, "
            "mais %s attendus.",
            segments, num_segments
        )
        return False
    
    logger.info("Vérifiant compatibilité : %s frames vs %s coupes source",
                num_frames, len(dicom_series))
    
    # Vérifier les dimensions spatiales
    if len(dicom_series) > 0:
        first_image = dicom_series[0]
        
        if (first_image.Rows, first_image.Columns) != (height, width):
            logger.error(
                "Dimensions spatiales incompatibles : "
                "mask (%d, %d) vs image (%d, %d)",
                height, width, first_image.Rows, first_image.Columns
            )
            return False
    
    logger.info("Compatibilité vérifiée")
    return True


def create_segment_description(segment_number: int = 1,
                               segment_label: str = "Tumor") -> SegmentDescription:
    """
    Create a segment description.

    Args:
        segment_number (int, optional): Number of the segment. Defaults to 1.

    Returns:
        SegmentDescription: Segment description.
    """
    algorithm_identification = AlgorithmIdentificationSequence(
        name="Tumor Segmentation",
        version="1.0",
        family=codes.cid7162.ArtificialIntelligence,
    )

    segment_description = SegmentDescription(
        segment_number=segment_number,
        segment_label=segment_label,
        segmented_property_category=codes.SCT.MorphologicallyAbnormalStructure,
        segmented_property_type=codes.SCT.Tumor,
        algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=algorithm_identification,
        tracking_uid=UID(),
        tracking_id="1",
    )

    return segment_description


def create_seg_object(
    source_images: List[Dataset],
    mask_array: np.ndarray,
    series_number: int,
    series_instance_uid: Optional[str] = None,
    num_segments: int = 1,
    num_frames: Optional[int] = None,
    segment_labels: Optional[List[str]] = None,
    series_description: str = "Automated Segmentation",

) -> Segmentation:
    """
    Create a DICOM SEG object from a binary mask and DICOM images.

    Args:
        source_images: List of DICOM datasets (source images)
        mask_array: Binary mask (2D/3D/4D, see prepare_mask_array_for_seg for formats)
        series_number: Number of the SEG series
        series_instance_uid: UID of the SEG series (auto-generated if None)
        num_segments: Number of segments (default 1)
        num_frames: Number of frames. REQUIRED if mask_array is 3D concatenated!
        segment_labels: Labels for each segment
        series_description: Description of the segmentation series

    Returns:
        Segmentation: DICOM SEG object.

    Raises:
        ValueError: If validation fails
        TypeError: If any of the input arguments are of the wrong type.
    """
    # Generate series_instance_uid if not provided
    if series_instance_uid is None:
        series_instance_uid = str(UID())

    # Déduire num_frames s'il n'est pas fourni
    if num_frames is None:
        if len(source_images) > 0:
            num_frames = len(source_images)
            logger.info(f"num_frames déduit de source_images : {num_frames}")
        else:
            raise ValueError(
                "num_frames doit être fourni ou source_images doit contenir des images"
            )

    # Validate input arguments
    if not isinstance(source_images, list) or not all(
        isinstance(img, Dataset) for img in source_images
    ):
        raise TypeError("source_images must be a list of pydicom.dcmread objects.")
    
    if not isinstance(mask_array, np.ndarray):
        raise TypeError("mask_array must be a numpy.ndarray.")
    
    if not isinstance(series_instance_uid, str):
        raise TypeError("series_instance_uid must be a string.")
    
    if not isinstance(series_number, int):
        raise TypeError("series_number must be integers.")

    # Préparer le masque avec les bonnes dimensions
    logger.info("Préparation du masque : shape initiale = %s", mask_array.shape)
    mask_array_prepared, shape_info = prepare_mask_array_for_seg(
        mask_array,
        num_frames=num_frames,
        num_segments=num_segments
    )
    num_frames_final, height, width, segments = shape_info
    logger.info("Masque préparé : shape finale = %s", mask_array_prepared.shape)


    # Check series compatibility
    if not check_series_compatibility(source_images, mask_array_prepared, num_segments):
        raise ValueError("Source images are not compatible with the binary mask.")

    # Créer les descriptions des segments
    if segment_labels is None:
        segment_labels = [f"Segment_{i+1}" for i in range(num_segments)]
    
    if len(segment_labels) != num_segments:
        raise ValueError(
            f"Number of segment labels ({len(segment_labels)}) does not match "
            f"num_segments ({num_segments})"
        )

    # Create segment descriptions
    segment_descriptions = []
    for i, label in enumerate(segment_labels):
        seg_desc = create_segment_description(
            segment_number=i+1,
            segment_label=label
        )
        segment_descriptions.append(seg_desc)
        logger.info("Segment %d : %s", i+1, label)

    logger.info("SeriesDescription :", series_description)


    # Create SEG object
    logger.info("Création du dataset DICOM SEG...")
    seg = Segmentation(
        source_images=source_images,
        pixel_array=mask_array,
        segmentation_type=SegmentationTypeValues.BINARY,
        segment_descriptions=segment_descriptions,
        series_instance_uid=series_instance_uid,
        series_number=series_number,
        instance_number=1,
        sop_instance_uid=UID(),
        manufacturer="Your Manufacturer",
        manufacturer_model_name="Your Model",
        software_versions="1.0",
        device_serial_number="Your Serial Number",
        series_description=series_description,
    )
    return seg


# Exemple d'utilisation
if __name__ == "__main__":
    ct_dicom_dir = Path("path/to/ct_series")
    pt_dicom_dir = Path("path/to/pt_series")
    mask_array = np.random.randint(
        0, 2, size=(len(list(pt_dicom_dir.iterdir())), 512, 512), dtype=np.uint8
    )  # Exemple de masque binaire
    series_instance_uid = "1.2.3.4.5.6.7.8.9.10"
    series_number = 100

    seg_object = create_seg_object(
        ct_dicom_dir, pt_dicom_dir, mask_array, series_instance_uid, series_number
    )

    # Sauvegarder l'objet DICOM SEG
    seg_object.save_as("path/to/seg.dcm")
