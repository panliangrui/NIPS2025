

<div align="center">
  <a href="(https://github.com/panliangrui/IJCAI25/blob/main/STAS.jpg)">
    <img src="https://github.com/panliangrui/IJCAI25/blob/main/STAS.jpg" width="800" height="400" />
  </a>

  <h1>STAS(solid nests, micropapillary clusters, single cells)</h1>

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

# SMILE: a Scale-aware Multiple Instance Learning Method for Multicenter STAS  Lung Lancer Histopathology Diagnosis

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

## Models
**SMILE**

  <a href="(https://github.com/panliangrui/IJCAI25/blob/main/flow.jpg)">
    <img src="https://github.com/panliangrui/IJCAI25/blob/main/flow.jpg" width="800" height="400" />
  </a>

Overall workflow of the proposed SMILE approach. We process the given bag through a joint feature representation module to transform them into instance features. These features are then processed through a scale-adaptive attention module to obtain scaled bag-level feature representations. Finally, the final STAS prediction results are obtained through the classifier g.



**Baseline MIL Methods**

This repository provides implementations and comparisons of various MIL-based methods for Whole Slide Image (WSI) classification.

- **Maxpooling**: Represents a slide by selecting the instance with the maximum activation, thereby mimicking the focus on the most prominent lesion.
- **Meanpooling**: Aggregates all instance features by computing their mean, thus treating each patch equally in the overall representation.
- **ABMIL**: Employs an attention mechanism to assign weights to instances, effectively prioritizing diagnostically relevant regions.
- **TransMIL**: A transformer-based MIL framework that leverages both morphological and spatial correlations among instances to enhance visualization, interpretability, and performance in WSI pathology classification.
- **CLAM-SB**: A clustering constraint-based attention MIL method that employs a single attention branch to aggregate instance features and generate a bag-level representation.
- **CLAM-MB**: The multi-branch version of the CLAM model, computing attention scores for each class separately to produce multiple unique bag-level representations.
- **DTFD-MIL**: Addresses the challenge of limited WSI samples in MIL by introducing pseudo-bags to virtually enlarge the bag count and implementing a double-tier framework that leverages an attention-based derivation of instance probabilities to effectively utilize intrinsic features.
- **ACMIL**: Mitigates overfitting by employing multiple branch attention and stochastic top-K instance masking to reduce attention value concentration and capture more discriminative instances in WSI classification.
- **ILRA**: Incorporates a pathology-specific Low-Rank Constraint for feature embedding and an iterative low-rank attention model for feature aggregation, achieving enhanced performance in gigapixel-sized WSI classification.
- **DGRMIL**: Models instance diversity by converting instance embeddings into similarities with predefined global vectors via a cross-attention mechanism and further enhances the diversity among these global vectors through positive instance alignment and a determinant point process-based diversified learning paradigm.

## Train Models
```markdown
python manage.py
```


## Datastes

- Only features of the histopathology image data are provided as the data has a privacy protection agreement.
```markdown
link: https://pan.baidu.com/s/1nm9IJ817UpMmc1h6d0zxPw?pwd=h45j password: h45j 
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
