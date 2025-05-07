

<div align="center">
  <a href="(https://github.com/panliangrui/NIPS2025/blob/main/liucheng.png)">
    <img src="https://github.com/panliangrui/NIPS2025/blob/main/liucheng.png" width="800" height="400" />
  </a>

  <h1>PathGene(NGS, Histopathology images)</h1>
  Flowchart of the collection and preprocessing of lung cancer patients’ histopathology images and NGS data.

  <p>
  Anonymous Author et al. is a developer helper.
  </p>

  <p>
    <a href="https://github.com/misitebao/yakia/blob/main/LICENSE">
      <img alt="GitHub" src="https://img.shields.io/github/license/misitebao/yakia"/>
    </a>
  </p>

  <!-- <p>
    <a href="#">Installation</a> | 
    <a href="#">Documentation</a> | 
    <a href="#">Twitter</a> | 
    <a href="https://discord.gg/zRC5BfDhEu">Discord</a>
  </p> -->

  <div>
  <strong>
  <samp>

[English](README.md)

  </samp>
  </strong>
  </div>
</div>

# PathGene: A benchmark for predicting driver gene mutations and exons based on a multicenter lung cancer histopathology image dataset

## Table of Contents

<details>
  <summary>Click me to Open/Close the directory listing</summary>

- [Table of Contents](#table-of-contents)
- [Feature Preprocessing](#Feature-Preprocessing)
- [Feature Extraction](#Feature-Extraction)
- [Models](#Train-models)
- [Train Models](#Train-models)
- [Datastes](#Datastes)
- [Installation](#Installation)
- [License](#license)

</details>

## Feature Preprocessing

Use the pre-trained model for feature preprocessing and build the spatial topology of WSI.

### Feature Extraction

Features extracted based on CTransPath.
Please refer to CTransPath: https://github.com/Xiyue-Wang/TransPath

Feature extraction code reference project: https://github.com/mahmoodlab/CLAM
```markdown
python create_patches_fp.py
```
```markdown
python extract_features_fp.py
```




**Baseline MIL Methods**

This repository provides implementations and comparisons of various MIL-based methods for Whole Slide Image (WSI) classification.

- **ABMIL**: is an attention mechanism-based deep multiple instance learning method, primarily designed to handle bag-level classification tasks (e.g., determining the malignancy of medical images), while preserving the ability to interpretably locate key instances. Its core framework is implemented in three stages: first, a neural network maps instances within a bag into low-dimensional embedded features; next, these instance features are aggregated in a permutation-invariant manner using dynamic attention weights combined with a gating mechanism; finally, label prediction is performed based on the aggregated bag embeddings. Compared to traditional methods, ABMIL achieves a balance between performance and efficiency through end-to-end training. It has demonstrated superior performance on datasets such as MNIST-BAGS, breast cancer, and colon cancer, particularly excelling in few-shot learning scenarios. Additionally, ABMIL can intuitively locate the critical instances (e.g., lesion areas in pathological images) that trigger the bag-level label through attention weights, fulfilling legal requirements for model interpretability in the medical domain.
- **TransMIL**: is a Transformer‐based correlated multiple instance learning framework tailored for weakly supervised WSI classification. It formally introduces the “correlated MIL” paradigm and provides a convergence proof, thereby overcoming the traditional MIL assumption of instance independence by explicitly modeling inter‐instance correlations via self‐attention. Architecturally, TransMIL employs ResNet50 to extract feature embeddings from WSI patches, which are then concatenated into a sequence for Transformer processing. Two parallel attention modules are incorporated to separately capture morphological and spatial information from the patches. A Pyramid Position Encoding Generator (PPEG) is devised to encode each patch’s two‐dimensional coordinates at multiple scales, with PPEG layers interposed between and following the Transformer encoder blocks to reinforce spatial dependency modeling. Before Transformer application, the patch sequence undergoes a “squaring” operation and a CLS token is prepended for global aggregation, allowing the resulting pooling matrix to contain both diagonal self‐attention weights and off‐diagonal cross‐instance correlation weights. The CLS token’s output is fed into a simple fully connected classification head to yield a bag‐level prediction, ensuring both training efficiency and high interpretability. 
- **CLAM-SB**: (Single-Branch Model) is a simplified variant of the CLAM framework specifically designed for binary classification tasks, such as distinguishing between tumor and normal tissue. The model identifies diagnostically significant sub-regions through a single attention module, treating regions with high attention scores as evidence for the positive class, while simultaneously using attention maps from other categories as supervisory signals to enforce cluster constraints. For example, in lymph node metastasis detection, CLAM-SB automatically focuses on micro-metastatic regions and optimizes the feature space via an instance-level clustering loss, encouraging a clear separation between positive and negative samples. Its main advantage lies in its lightweight architecture, making it well-suited for scenarios with limited data and mutually exclusive classes.
- **CLAM-MB**: (Multi-Branch Model) is designed for multi-subtype classification tasks such as differentiating subtypes of renal cell carcinoma or non-small cell lung cancer and employs a parallel attention branch architecture. Each branch corresponds to a specific class, independently computing class-specific attention weights and generating slide-level feature representations. For instance, in the classification of renal cell carcinoma, the three branches of CLAM-MB are dedicated to clear cell carcinoma, papillary carcinoma, and chromophobe carcinoma, respectively. A competitive attention mechanism is used to select key morphological features for each subtype, such as cytoplasmic clearing or papillary structures. Additionally, cross-branch clustering constraints are introduced to enforce feature-space separation among high-attention regions of different subtypes, thereby improving classification specificity and interpretability.
- **DTFD-MIL**: addresses the paucity of WSIs by virtually expanding each slide into multiple $pseudo-bags$, each inheriting the parent slide’s label. In Tier 1, an attention-based MIL model is trained on these pseudo-bags to produce initial bag-level predictions and to derive instance-level probabilities via Grad-CAM under the AB-MIL framework. However, since some pseudo-bags from positive slides may contain no true positives, Tier 2 distills representative feature vectors from each pseudo-bag—using strategies like selecting the top-scoring instance (MaxS) or both top and bottom (MaxMinS)—and retrains a second attention-based MIL model on the original slide bags, effectively correcting mislabeled pseudo-bags and enhancing robustness. This two-stage distillation cascade yields more discriminative slide-level embeddings, producing substantial accuracy gains over single-tier MIL on CAMELYON-16 and TCGA lung cancer cohorts while maintaining strong generalization across datasets.
- **DSMIL**: (Dual-Stream Multiple Instance Learning Network) is a weakly supervised deep learning model designed for WSI classification. Its key innovations include a dual-stream aggregation architecture, self-supervised feature learning, and a multi-scale feature fusion mechanism. In DSMIL, a WSI is treated as a bag containing multiple instances (image patches), and the model is trained using only slide-level labels. To enhance feature representation, DSMIL employs a self-supervised contrastive learning strategy based on the SimCLR framework to pretrain the feature extractor. This allows the model to learn robust patch-level representations without the need for local annotations, while also reducing memory consumption when handling large bags. Furthermore, the model incorporates a pyramid-style multi-scale fusion approach that concatenates features extracted from image patches at different magnification levels. This design preserves global tissue architecture while capturing detailed cellular morphology across scales. Additionally, by sharing low-level features across scales, DSMIL imposes spatial continuity constraints on attention weights, which improves both classification accuracy and the precision of lesion localization.
- **IBMIL**: reframes MIL as a causal inference problem to eliminate spurious correlations induced by slide‐level contextual priors.  It begins by constructing a Structural Causal Model (SCM) that identifies the bag context (e.g., staining protocol, scanner artifacts) as a confounder between the bag’s instances and its label.  During training, IBMIL applies Pearl’s backdoor adjustment via an “interventional” reweighting of instance contributions, effectively simulating $\mathrm{do}(\text{bag})$ interventions to block confounding paths and learn de‐biased bag‐level predictors.  This interventional training is orthogonal to existing MIL aggregation schemes—attentive, pooling, or transformer‐based—and thus can be layered on top of any feature extractor or aggregator to consistently boost performance.  Empirically, IBMIL demonstrates state-of-the-art slide-level classification on CAMELYON16 by reducing false positives caused by background artifacts, while maintaining competitive computational efficiency and retaining interpretability through its intervention‐guided instance scores.
- **HIPT**: is a two-stage, multi-scale MIL framework that models WSIs via a coarse-to-fine pyramid of patch resolutions. In the first stage, the slide is partitioned into large, low-resolution tiles whose embeddings are processed by a Transformer encoder to capture global tissue architecture. The resulting “coarse” CLS representations are then distilled down to guide the second stage, where smaller, high-resolution tiles are encoded by a separate Transformer; here, the coarse CLS tokens act as teachers, enforcing consistency via a distillation loss so that fine-scale features remain aware of the overall context. To fuse spatial hierarchy, HIPT employs overlapping windowed self-attention and incorporates positional encodings that respect both tile location and pyramid level. This hierarchical distillation not only accelerates convergence by allowing the coarse model to provide robust initialization for the fine model but also yields richer, multi-scale slide embeddings that improve classification accuracy and interpretability—particularly in low-data or highly heterogeneous pathology cohorts.
- **MHIM**: streamlines attention-based MIL by explicitly mining “hard” instances to improve robustness. A momentum “teacher” model scores patches, identifying those of intermediate difficulty, while a “student” model is trained on masked subsets that include both easy and hard examples. The teacher’s weights update via exponential moving average of the student, guiding the student to focus on diagnostically challenging patches. This mask-based instance reweighting yields more balanced learning, reduces overfitting, and consistently outperforms standard MIL baselines on WSI classification tasks.
- **RRT_MIL**: augments any backbone–aggregator pipeline by re-embedding patch features on-the-fly through a lightweight Transformer module. After extracting patch embeddings (e.g., with ResNet-50), it partitions them into local regions and applies a small Transformer—with relative or convolutional position encodings—to refine those embeddings before MIL aggregation. During training, regions (1D or 2D blocks) are ranked by their refined scores and reweighted: top regions receive higher attention while lower-ranked ones are downplayed. This plug-and-play R²T block enables end-to-end fine-tuning of both the backbone and MIL head, yielding large AUC gains across CAMELYON-16, TCGA-NSCLC, and other cohorts compared to static-feature MIL methods.
- **Patch-GCN**: treats a WSI as a 2D point cloud where each patch becomes a graph node and edges connect spatially adjacent patches via k-nearest neighbors, explicitly encoding tissue architecture and local context for downstream tasks.  Patch features are first extracted with a CNN backbone (e.g., ResNet-50) and serve as initial node embeddings for a multi-layer graph convolutional network (GCN) that performs message passing to aggregate both morphological characteristics and neighborhood interactions.  To capture structures at multiple scales, a hierarchical clustering strategy groups patches globally and locally—forming subgraphs that preserve macro-level distribution patterns as well as micro-level histologic details—and these subgraphs are processed sequentially or in parallel by the GCN.  After GCN layers, a slide-level representation is obtained via a global readout operation (mean, max, or attention pooling), which is then fed into a classifier or survival-prediction head.  
- 
## Train Models
```markdown
python manage.py
```


## Datastes

- Only features of the histopathology image data are provided as the data has a privacy protection agreement.
```markdown
link: https://pan.baidu.com/s/1zpt7D_XNgqZpLnUyOmtkgA?pwd=8yn6 password: 8yn6
```
- We provide clinical data on STAS patients, including patient age, gender, stage and protein level expression data.
Please contact the corresponding author or first author by email.
- STAS_CSU: From April 2020 to December 2023, we selected 356 patients at the Second Xiangya Hospital who underwent pulmonary nodule resection and were diagnosed with lung cancer (particularly those with STAS) to form a retrospective lung cancer cohort. We comprehensively collected each patient's clinical and pathological data, including age, tumor size, lymph node metastasis, distant metastasis, clinical stage, recurrence, and survival status. Two experienced pathologists independently reviewed the pathology data for every patient, including frozen and paraffin H\&E slides as well as immunohistochemical (IHC) slides, confirming the presence or absence of STAS, the specific pathological subtype of any disseminated foci, the detailed histological subtype of lung cancer, and the expression of key proteins (PD-L1, TP53, Ki-67, and ALK). Within this cohort, there were 150 non-STAS patients and 206 STAS patients. Each patient's tumor specimen was sectioned by pathologists into multiple paraffin blocks, each corresponding to multiple H\&E slides. In total, we collected and scanned 1{,}290 frozen and paraffin slides and 1{,}436 IHC slides. Of these, 247 frozen slides comprised 81 STAS and 158 non-STAS histopathological images, while 1{,}043 paraffin slides contained 585 STAS and 436 non-STAS images. These slides were divided into two sets for internal validation and testing of our model. Furthermore, this dataset includes survival time and status for all patients.

- STAS_ZZU: From June 2023 to the present, the Affiliated Tumor Hospital of Zhengzhou University and Henan Cancer Hospital collected 100 paraffin sections from 20 STAS patients. According to the inclusion and exclusion criteria set by pathologists, 91 histopathological images were retained. All WSIs were annotated by pathologists to indicate STAS presence, dissemination focus type, and tumor type. Among these WSIs, 60 were STAS and 31 were non-STAS, forming a small-scale STAS dataset.

- STAS_TCGA:} We downloaded relevant LUAD WSIs from the TCGA website {https://portal.gdc.cancer.gov/}. Based on our inclusion and exclusion standards, we collected 506 paraffin sections from an unspecified number of patients. All WSIs underwent cross-verification by two experienced pathologists to determine STAS status, type of dissemination foci, and tumor type. Finally, following the inclusion and exclusion criteria, the STAS\_TCGA dataset consists of 117 STAS WSIs and 115 non-STAS WSIs, along with corresponding patient survival times and statuses.

- STAS_CPTAC: We obtained 1{,}139 paraffin sections from the CPTAC{https://www.cancerimagingarchive.net/collection/cptac-luad/} . In accordance with our inclusion and exclusion rules, 53 WSIs were labeled as STAS and 47 were labeled as non-STAS. These images were subsequently used to assess the generalizability of our model.

## Installation
- Linux (Tested on Ubuntu 18.04)
- NVIDIA GPU (Tested on a single Nvidia GeForce RTX 4090)
- Python (3.7.11), h5py (2.10.0), opencv-python (4.1.2.30), PyTorch (1.10.1), torchvision (0.11.2), pytorch-lightning (1.5.10).


## License
If you need the original histopathology image slides, please send a request to our email address. The email address will be announced after the paper is accepted. Thank you!

[License MIT](../LICENSE)
