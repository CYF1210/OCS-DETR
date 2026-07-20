# OCS-DETR: Optimal Transport and Coarse-to-Fine Saliency Fusion for Interdependent Video Moment Retrieval and Highlight Detection


## Overview

This repository contains the official implementation of **OCS-DETR**, a DETR-based framework for joint **Video Moment Retrieval (MR)** and **Highlight Detection (HD)**.

Unlike previous DETR-based methods that independently predict moment localization and highlight scores, OCS-DETR introduces task reciprocity between MR and HD, allowing the two tasks to mutually refine each other.

Our framework is built upon QD-DETR and introduces three major improvements:

- **Optimal Transport Aligner (OTAligner)**:
  A gated optimal transport module that performs token-level video-text alignment before Transformer reasoning.

- **Visual Refinement Fuser (VRFuser)**:
  A cross-modal fusion module that strengthens video-query interaction through bidirectional attention.

- **Interdependent Saliency Fusion**:
  - Saliency-Guided Refinement (SGR): HD helps MR by suppressing irrelevant video clips.
  - Coarse-to-Fine Moment Attention Fusion (CMAF): MR helps HD by providing temporal priors.


## Framework Overview

<p align="center">
<img src="figure.png" width="800">
</p>


## Main Contributions

- Introduce a reciprocal DETR framework where Highlight Detection helps Moment Retrieval and Moment Retrieval helps Highlight Detection.

- Propose a gated Optimal Transport Alignment module to improve fine-grained video-text semantic correspondence.

- Design VRFuser to enhance cross-modal interaction before Transformer encoding.

- Develop coarse-to-fine saliency fusion to refine highlight prediction using retrieved moment proposals.

- Improve training stability with focal loss and remove unnecessary negative-pair forward computation.


## Prerequisites

### Environment

Python >= 3.8

PyTorch >= 1.x


Install dependencies:

```bash
pip install -r requirements.txt
