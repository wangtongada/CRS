# CRS

#README

This repository contains datasets used in the paper "Causal Rule Sets for Identifying Subgroups with Enhanced Treatment Effects".

### CITATION
Wang, Tong, and Cynthia Rudin. "Causal rule sets for identifying subgroups with enhanced treatment effect." arXiv preprint arXiv:1710.05426 (2017).

### FILES
Folder "datasets" contains three datasets used in Section 6 in the paper, young voter turnout data, coupon data (a more detailed description of coupon data can be found [here](https://jmlr.org/papers/v18/16-003.html)), and a subset of crowdfunding data. Specifically, some sensitive features in the crowdfunding data have been modified by shifting the distribution. Other datasets in Section 6 are publicly available, and the links to those datasets can be found in the Online Supplement 2 of the paper.

Simulations.ipynb contains codes for generating random datasets for Section 5 in the paper, which includes generating true subgroups with rules, adding random noise, generating linear subgroups, and generating non-linear subgroups.

