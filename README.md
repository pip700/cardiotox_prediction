

# [**CardiotoxPred**](https://cardiotoxprediction.streamlit.app/): Cardiotoxicity Prediction Using Deep Learning

**CardiotoxPred** is a GNN-based tool designed to predict cardiotoxicity of chemical compounds based on their molecular structure, provided in SMILES format. The tool not only performs classification (Blocker vs. Non-blocker) but also delivers **atom-level and bond-level interpretability** via importance scores and visualizations.

**CardiotoxPred** is available as both a user-friendly [web application](https://cardiotoxprediction.streamlit.app/) built with Streamlit and a Docker container, allowing seamless access for both casual users and developers. The web interface enables quick predictions and visualizations without any setup, while the Docker container provides a portable and reproducible environment suitable for local or large-scale deployment.

---

## 📚 Table of Contents

* [Features](#-features)
* [Installation](#-installation)
* [Usage `CardiotoxPred`](#-usage)
* [Output Results](#-output-results)
* [Troubleshooting](troubleshooting)
* [Contributors](#-contributors)
* [Citation](#-citation)

---

## 🚀 Features

* ✅ Predicts whether a molecule is a **Blocker** or **Non-blocker**
* ⚙️ Supports both **CPU** and **GPU** inference
* 🔍 Provides **atom- and bond-level importance scores**
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

All dependencies are pre-installed in the Docker image. Only need to run Docker.

---

## 🧪 Usage

### 2. Run the Docker Container

Navigate to your working directory and run the container:

#### CPU-only version:

``` bash
sudo docker run -it --rm -v ${PWD}:/workspace ghcr.io/pip700/cardiotoxpred:cpu
```

#### GPU-enabled version:

``` bash
sudo docker run --gpus all -it --rm -v ${PWD}:/workspace ghcr.io/pip700/cardiotoxpred:gpu
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
* `Atom-level_importance.csv`: Importance scores for each atom
* `Bond-level_importance.csv`: Importance scores for each bond
* `Atom & Bond-level.png`: Visual representation of importance scores

---



## 🛠️ Troubleshooting

* **Docker not found**: Ensure Docker is installed and added to PATH
* **Permission denied**: Try running Docker with `sudo`
* **GPU not detected**: Ensure NVIDIA drivers and Docker's GPU support are installed correctly

---

## 👥 Contributors

* [pip700](https://github.com/pip700)



---

## 📑 Citation

If you use **CardiotoxPred** in your research or publication, please consider citing the following paper:

```bibtex
@article{,
  author    = {Dhairiya Agarwal, Anju Sharma, and Prabha Garg},
  title     = {CardiotoxPred: GNN-Based Classification Model for Predicting Cardiac Toxicity: Kav, Cav, Nav},
  journal   = {...},
  volume    = {},
  year      = {2025},
  url       = {https://pubmed.ncbi.nlm.nih.gov/...},
  doi       = {...},
  issn      = {}
}
```
