# AAE_STGNN: Area-Specific Autoencoder Spatial–Temporal Graph Neural Networks for Opioid Overdose Death Prediction

## Abstract
Objective: Ohio has been severely impacted by the opioid crisis, with OD death rates exceeding
national averages. Accurate OD death prediction supports proactive prevention and treatment
allocation. Existing methods often focus on ZIP Code Tabulation Area (ZCTA)–level prediction for
small-area resource allocation; however, performance at this resolution is poor due to substantial
fluctuations in OD death counts, which introduce noise. This raises a critical methodological question:
what is the optimal population threshold for OD death prediction that balances predictive accuracy
with geographic resolution?
Materials and Methods: We perform a theoretical analysis of variance and error bounds to establish
the minimal population required for robust prediction. Building on this analysis, we propose an Area-
specific AutoEncoder Spatial--Temporal Graph Neural Network (AAE-STGNN) framework for opioid
OD death count prediction using Urine Drug Test (UDT) data as dynamic features and Social
Determinants of Health (SDoH) as static features. The framework consists of two key components: (1)
an Area-Specific Autoencoder (AAE), which learns latent spatial representations while incorporating
the minimal population threshold, and (2) a Spatial-Temporal Graph Neural Network (STGNN), which
models geographic adjacency between areas and dynamic features across time.
Results: Empirical evaluations demonstrate that AAE-STGNN outperforms state-of-the-art
approaches, achieving improved accuracy and robustness. We also provide the OD death count trend
estimation to support public health decision-making.
Discussion and Conclusion: These findings underscore the importance of selecting an optimal spatial
granularity and leveraging spatial–temporal modeling techniques for data-driven public health
surveillance and targeted intervention in the opioid crisis.


## Dataset
Socioal determinants of health (SDoH) data are publicly accessible from the U.S. Census Bureau’s American Community Survey (ACS). Access to individual-level urine drug test (UDT) data requires appropriate data use agreements and can be requested from the corresponding author in collaboration with Millennium Health staff.



<!-- ## Cite
If our work is helpful to you, please cite:
<!-- 
 ```html 
 @INPROCEEDINGS{9669358,  
  author={Chen, Xianhui and Chen, Ying and Ma, Wenjun and Fan, Xiaomao and Li, Ye},  
  booktitle={2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},  
  title={SE-MSCNN: A Lightweight Multi-scaled Fusion Network for Sleep Apnea Detection Using Single-Lead ECG Signals},  
  year={2021},  
  volume={}, 
  number={},  
  pages={1276-1280},  
  doi={10.1109/BIBM52615.2021.9669358}}
  ```
   
Recently, another manuscript has published in Knowledge-Based System, which provides ablation experiments and computation complexity analysis. -->

## Email
If you have any questions, please email to: [chen.11773@osu.edu](mailto:chen.11773@osu.edu)