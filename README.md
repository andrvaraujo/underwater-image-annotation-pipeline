# Underwater Image Annotation Pipeline

A semi-automatic pipeline developed at [INESC TEC](https://www.inesctec.pt/) as part of the Integrated Project at FEUP (2026).

The project supports the development of **semantic communications for underwater environments**. Because underwater communication channels have limited bandwidth and high latency, transmitting raw images or video can be inefficient. The broader goal is therefore to transmit relevant semantic information about a scene instead.

This repository focuses on the dataset side of that problem: generating and reviewing annotations for underwater images.

## Pipeline

```text
Underwater images
        ↓
    MobileSAM
Instance segmentation
        ↓
  Mask filtering
        ↓
      CLIP
Zero-shot classification
        ↓
 COCO annotations
        ↓
      CVAT
Human review
```

The pipeline was implemented in Python and uses MobileSAM, CLIP, OpenCV, PyTorch, COCO-format annotations and CVAT. Image augmentation was also explored separately with Albumentations.

## Files

- `pipeline_CLIP.py` — segmentation, zero-shot classification, COCO annotation generation and CVAT integration
- `augmentation.py` — image augmentation script
- `requirements.txt` — Python dependencies
- `.gitignore` — excludes local datasets, model checkpoints and generated files

## Results

The pipeline worked best on images with clear water, good illumination and a stable camera. Performance degraded with turbidity, motion blur, strong reflections, low contrast, transparent objects and small objects.

The experiments also showed that image acquisition quality had a stronger effect on performance than further parameter tuning or basic augmentation.

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

`pipeline_CLIP.py` additionally requires:

- a MobileSAM checkpoint (`mobile_sam.pt`);
- a local CVAT instance;
- input images in the configured `images/` folder.

The dataset, model checkpoint, generated annotations and other local project data are intentionally not included in this repository.

## Internship

Developed by **André Carvalho Araújo** during an internship at **INESC TEC**, as part of the FEUP Integrated Project, 2026.

The full methodology, experiments and limitations are documented in the project report.
