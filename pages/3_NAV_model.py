import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = '42'

import random
import numpy as np
import torch
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import base64

# Set all random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Enable deterministic algorithms
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Rest of imports
from torch_geometric.data import Data
from torch_geometric.nn import AttentiveFP
from torch_geometric.utils import from_smiles
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import SimilarityMaps
import warnings
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('ggplot')

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AttentiveFPModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attentivefp = AttentiveFP(
            in_channels=9, hidden_channels=384, out_channels=1,
            edge_dim=3, num_layers=6, num_timesteps=3, dropout=0.1
        )

    def forward(self, x, edge_index, edge_attr, batch):
        return self.attentivefp(x, edge_index, edge_attr, batch)

# load model with default location
model_path = "models/Nav.pth"

@st.cache_resource
def load_model(model_path):
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model = AttentiveFPModel().to(device)

        # Filter out unexpected keys from state_dict
        model_state_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                         if k in model_state_dict and v.size() == model_state_dict[k].size()}

        # Update model's state_dict with filtered pretrained_dict
        model_state_dict.update(pretrained_dict)
        model.load_state_dict(model_state_dict)

        model.eval()
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def process_smiles(smiles, model):
    try:
        # First validate the SMILES can be converted to a molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.warning(f"Invalid SMILES: {smiles}")
            return None

        model.eval()
        with torch.no_grad():
            data = from_smiles(smiles).to(device)

            # Check for empty tensors
            if data.num_nodes == 0 or data.x is None or data.x.nelement() == 0:
                st.warning(f"Empty graph generated for SMILES: {smiles}")
                return None

            data.x = data.x.float()
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

            # Additional check for edge attributes
            if data.edge_attr is not None and data.edge_attr.nelement() == 0:
                data.edge_attr = None

            pred = torch.sigmoid(model(data.x, data.edge_index, data.edge_attr, batch))

            # Ensure prediction is not empty
            if pred.nelement() == 0:
                st.warning(f"Empty prediction for SMILES: {smiles}")
                return None

            pred_value = pred.item()

        return {
            'smiles': smiles,
            'prediction': pred_value,
            'predicted_class': 'Blocker' if pred_value > 0.5 else 'Non-blocker',
            'node_features': data.x.cpu().numpy(),
            'edge_features': data.edge_attr.cpu().numpy() if data.edge_attr is not None else None,
            'graph_data': data
        }
    except Exception as e:
        st.error(f"Error processing SMILES {smiles}: {str(e)}")
        return None

def visualize_importance(smiles, node_weights, bond_weights):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None, None

    AllChem.Compute2DCoords(mol)
    num_atoms = mol.GetNumAtoms()

    # Process node weights
    node_weights = [float(w) for w in node_weights[:num_atoms]]
    min_node = min(node_weights)
    max_node = max(node_weights)
    node_weights_normalized = [(w - min_node)/(max_node - min_node + 1e-8) for w in node_weights]

    # Create atom importance dataframe
    atom_importance_df = pd.DataFrame({
        'Atom_Index': range(num_atoms),
        'Importance': node_weights,
        'Normalized_Importance': node_weights_normalized,
        'Element': [atom.GetSymbol() for atom in mol.GetAtoms()]
    })

    # Atom importance visualization
    drawer = Draw.MolDraw2DCairo(600, 600)
    SimilarityMaps.GetSimilarityMapFromWeights(
        mol, node_weights_normalized,
        colorMap='bwr', contourLines=10,
        draw2d=drawer
    )
    drawer.FinishDrawing()
    atom_img = Image.open(BytesIO(drawer.GetDrawingText()))

    # Initialize bond importance variables
    bond_importance_df = None
    bond_img = None

    # Process bond weights if they exist
    if bond_weights is not None and len(bond_weights) > 0:
        try:
            bond_weights = np.array(bond_weights).flatten()  # Ensure bond_weights is 1D array
            num_bonds = mol.GetNumBonds()

            if len(bond_weights) >= num_bonds:
                # Create bond importance dataframe
                bond_info = []
                for bond in mol.GetBonds():
                    idx = bond.GetIdx()
                    imp = float(bond_weights[idx]) if idx < len(bond_weights) else 0.5
                    bond_info.append({
                        'Bond_Index': idx,
                        'Atom1': bond.GetBeginAtomIdx(),
                        'Atom2': bond.GetEndAtomIdx(),
                        'Bond_Type': str(bond.GetBondType()),
                        'Importance': imp
                    })
                bond_importance_df = pd.DataFrame(bond_info)

                # Bond importance visualization
                drawer = Draw.MolDraw2DCairo(600, 600)
                op = Draw.MolDrawOptions()
                op.addBondIndices = True
                drawer.SetDrawOptions(op)

                highlight_bonds = []
                bond_colors = {}
                min_bond = min(bond_weights)
                max_bond = max(bond_weights)

                for bond in mol.GetBonds():
                    idx = bond.GetIdx()
                    if idx < len(bond_weights):
                        imp = float(bond_weights[idx])
                    else:
                        imp = 0.5
                    imp_normalized = (imp - min_bond)/(max_bond - min_bond + 1e-8)
                    highlight_bonds.append(idx)
                    bond_colors[idx] = (float(imp_normalized), 0.0, float(1-imp_normalized))

                highlight_atoms = list(range(mol.GetNumAtoms()))

                drawer.DrawMolecule(
                    mol,
                    highlightAtoms=highlight_atoms,
                    highlightBonds=highlight_bonds,
                    highlightBondColors=bond_colors
                )
                drawer.FinishDrawing()
                bond_img = Image.open(BytesIO(drawer.GetDrawingText()))

        except Exception as e:
            st.warning(f"Error processing bond weights: {e}")

    return atom_img, bond_img, atom_importance_df, bond_importance_df

def get_smiles_column(df):
    """Helper function to identify SMILES column"""
    possible_names = ['smiles', 'SMILES', 'Smiles', 'structure', 'STRUCTURE']
    for col in df.columns:
        if col in possible_names:
            return col
    return None

def get_table_download_link(df, filename):
    """Generates a link allowing the data in a given panda dataframe to be downloaded"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def main():
    st.set_page_config(page_title="Nav_Predictor", layout="wide") # Note: This title is from KAV, might want to change for NAV
    st.title("Nav_Predictor - Nav Channel Blocker Classification") # Changed title for clarity
    st.markdown("""
    This app predicts whether a molecule is a Nav channel blocker using an AttentiveFP model.
    """)

    # Load model (cached)
    model = load_model(model_path)
    if model is None:
        st.stop()

    # Input options
    input_option = st.radio("Input type:", ("Single SMILES", "CSV File"))

    if input_option == "Single SMILES":
        smiles = st.text_input("Enter SMILES string:", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
        smiles_list = [smiles.strip()] if smiles else []
        input_df = None
    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            smiles_col = get_smiles_column(input_df)

            if smiles_col is None:
                st.warning("Could not automatically detect SMILES column. Please select it from the dropdown.")
                smiles_col = st.selectbox("Select SMILES column:", input_df.columns)

            smiles_list = input_df[smiles_col].tolist()
            st.success(f"Found {len(smiles_list)} SMILES strings in column '{smiles_col}'")
        else:
            smiles_list = []
            input_df = None

    if st.button("Predict") and smiles_list:
        progress_bar = st.progress(0)
        results = []

        for i, smiles in enumerate(smiles_list):
            progress_bar.progress((i + 1) / len(smiles_list))

            with st.spinner(f"Processing molecule {i+1}/{len(smiles_list)}: {smiles[:50]}..."):
                res = process_smiles(smiles, model)
                if res is not None:
                    # Calculate node and bond importances
                    node_imp = res['node_features'].mean(axis=1)
                    bond_imp = res['edge_features'].mean(axis=1) if res['edge_features'] is not None else None

                    # Generate visualizations and get importance dataframes
                    atom_img, bond_img, atom_imp_df, bond_imp_df = visualize_importance(
                        smiles,
                        node_imp,
                        bond_imp
                    )

                    res['atom_img'] = atom_img
                    res['bond_img'] = bond_img
                    res['atom_importance_df'] = atom_imp_df
                    res['bond_importance_df'] = bond_imp_df
                    results.append(res)

        if results:
            st.success("Prediction completed successfully!")

            # Display results
            for i, res in enumerate(results):
                st.subheader(f"Molecule {i+1}: {res['smiles']}")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Prediction:** {res['prediction']:.4f}")
                    st.markdown(f"**Class:** {res['predicted_class']}")

                    if res['atom_img']:
                        st.image(res['atom_img'], caption="Atom Importance", use_container_width=True) # Changed
                    if res['bond_img']:
                        st.image(res['bond_img'], caption="Bond Importance", use_container_width=True) # Changed

                with col2:
                    if res['atom_importance_df'] is not None:
                        st.dataframe(res['atom_importance_df'])

                    if res['bond_importance_df'] is not None:
                        st.dataframe(res['bond_importance_df'])

            # Download options
            st.subheader("Download Results")

            # Prepare data for download
            predictions = []
            atom_importances = []
            bond_importances = []

            for i, res in enumerate(results):
                pred_dict = {
                    'Molecule_Index': i,
                    'SMILES': res['smiles'],
                    'Prediction': res['prediction'],
                    'Predicted_Class': res['predicted_class']
                }

                # Add all columns from input file if available
                if input_df is not None and i < len(input_df):
                    for col in input_df.columns:
                        if col != 'SMILES' and col != 'smiles':  # Avoid duplicate SMILES column
                            pred_dict[col] = input_df.iloc[i][col]

                predictions.append(pred_dict)

                if res['atom_importance_df'] is not None:
                    df = res['atom_importance_df'].copy()
                    df.insert(0, 'Molecule_Index', i)
                    atom_importances.append(df)

                if res['bond_importance_df'] is not None:
                    df = res['bond_importance_df'].copy()
                    df.insert(0, 'Molecule_Index', i)
                    bond_importances.append(df)

            # Create download buttons
            if predictions:
                predictions_df = pd.DataFrame(predictions)
                st.markdown(get_table_download_link(predictions_df, "predictions.csv"), unsafe_allow_html=True)

            if atom_importances:
                atom_imp_df = pd.concat(atom_importances)
                st.markdown(get_table_download_link(atom_imp_df, "atom_importance.csv"), unsafe_allow_html=True)

            if bond_importances:
                bond_imp_df = pd.concat(bond_importances)
                st.markdown(get_table_download_link(bond_imp_df, "bond_importance.csv"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
