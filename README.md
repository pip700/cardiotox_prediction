

# [**CardiotoxPred**]: Cardiotoxicity Prediction Using Deep Learning

**CardiotoxPred** is a GNN-based tool designed to predict cardiotoxicity of chemical compounds based on their molecular structure, provided in SMILES format. The tool not only performs classification (Blocker vs. Non-blocker) but also delivers **atom-level and bond-level interpretability** visualizations.

---

## 📚 Table of Contents

* [Features](#-features)
* [Installation](#-installation)
* [Usage `CardiotoxPred`](#-usage)
* [Output Results](#-output-results)
* [Citation](#-citation)

---

## 🚀 Features
* ✅ Predicts whether a molecule is a **Blocker** or **Non-blocker**
* 💾 Saves prediction results and visualizations with **timestamped directories**
* 🖼️ Outputs visualizations to help interpret model predictions

---

## 📦 Installation

### 1. Install Docker

#### For Windows
Powershell
``` powershell
wsl --install
```
wsl
``` wsl
sudo apt-get update
sudo apt-get install docker.io -y
```

#### For Linux
bash
``` bash
sudo apt-get update
sudo apt-get install docker.io -y
```


#### 📦 Dependencies

***All dependencies are pre-installed in the Docker image. Only need to run Docker.***

---

## 🧪 Usage

### 2. Run the Docker Container

Navigate to your working directory and run the container:


``` bash
sudo docker run -it --rm -v ${PWD}:/workspace ghcr.io/pip700/cardiotoxpred:v2
```


### 3. Input File Selection

Once inside the Docker container: Select input format: Either `1`: SMILES or `2`: CSV file......
  
* Enter `1` to choose **SMILE** format, Provide the smile string, e.g.:

  ```
  COC1=CC=C(C=C1)CCN2CCC(CC2)NC3=NC4=CC=CC=C4N3CC5=CC=C(C=C5)F
  ```
* Enter `2` to choose **CSV** format, Provide the filename, e.g.:

  ```
  samples.csv
  ```

> The CSV file **must** contain a column with valid SMILES strings.

---

## 📤 Output Results

All output files are saved in a **timestamped directory** created in your working folder.

### Output Includes:

* `Prediction.csv`: Predicted labels and probabilities (Blocker / Non-blocker)
* `Atom & Bond-level.png`: Visual representation of importance for the prediction

---


## 📑 Citation

If you use **CardiotoxPred** in your research or publication, please consider citing the following paper:

```bibtex
@article{,
  author    = {Dhairiya Agarwal, Anju Sharma, and Prabha Garg},
  title     = {Graph-Based Classification with GNN-Explainer for Predicting Cardiac Toxicity Associated with Multi-Ion Channel Blockers},
  journal   = {Chemical Research in Toxicology},
  volume    = {39},
  year      = {2026},
  url       = {https://pubs.acs.org/doi/10.1021/acs.chemrestox.5c00369},
  doi       = {https://doi.org/10.1021/acs.chemrestox.5c00369},
  issn      = {4}
}
```
