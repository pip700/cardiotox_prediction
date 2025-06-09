# project/pages/utils.py

import os
import random
import numpy as np
import torch
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import base64
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import SimilarityMaps
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def set_global_configs():
    """Sets environment variables, random seeds, and deterministic algorithms."""
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = '42'
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    plt.style.use('ggplot')

# Call once at the start of the application
set_global_configs()

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def visualize_importance(smiles, node_weights, bond_weights, index=None):
    """
    Generates molecular importance visualizations (atom and bond) and corresponding dataframes.
    Args:
        smiles (str): SMILES string of the molecule.
        node_weights (np.array): Array of atom importance weights.
        bond_weights (np.array, optional): Array of bond importance weights.
        index (int, optional): Molecule index, used for adding to DataFrames.
    Returns:
        tuple: (atom_img (PIL.Image), bond_img (PIL.Image), atom_importance_df (pd.DataFrame), bond_importance_df (pd.DataFrame))
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.warning(f"Could not create RDKit molecule from SMILES: {smiles}")
        return None, None, None, None

    AllChem.Compute2DCoords(mol)
    num_atoms = mol.GetNumAtoms()

    atom_img, bond_img = None, None
    atom_importance_df, bond_importance_df = None, None

    # --- Atom Importance ---
    if node_weights is not None and len(node_weights) > 0:
        node_weights = np.array(node_weights).flatten()[:num_atoms] # Ensure correct length
        min_node = np.min(node_weights)
        max_node = np.max(node_weights)
        # Add a small epsilon to avoid division by zero if all weights are the same
        node_weights_normalized = [(w - min_node) / (max_node - min_node + 1e-8) for w in node_weights]

        atom_data = {
            'Atom_Index': range(num_atoms),
            'Importance': node_weights,
            'Normalized_Importance': node_weights_normalized,
            'Element': [atom.GetSymbol() for atom in mol.GetAtoms()]
        }
        if index is not None:
            atom_data['Molecule_Index'] = index
        atom_importance_df = pd.DataFrame(atom_data)

        try:
            drawer = Draw.MolDraw2DCairo(600, 600)
            SimilarityMaps.GetSimilarityMapFromWeights(
                mol, node_weights_normalized,
                colorMap='bwr', contourLines=10,
                draw2d=drawer
            )
            drawer.FinishDrawing()
            atom_img = Image.open(BytesIO(drawer.GetDrawingText()))
        except Exception as e:
            st.warning(f"Error generating atom importance map for SMILES {smiles}: {e}")
            atom_img = None

    # --- Bond Importance ---
    if bond_weights is not None and len(bond_weights) > 0:
        bond_weights = np.array(bond_weights).flatten()
        num_bonds = mol.GetNumBonds()

        if len(bond_weights) >= num_bonds:
            bond_info = []
            min_bond = np.min(bond_weights)
            max_bond = np.max(bond_weights)

            highlight_bonds = []
            bond_colors = {}

            for bond in mol.GetBonds():
                idx = bond.GetIdx()
                # Use the weight for this bond if available, otherwise default to 0.5 (neutral)
                imp = float(bond_weights[idx]) if idx < len(bond_weights) else 0.5
                imp_normalized = (imp - min_bond) / (max_bond - min_bond + 1e-8)

                bond_data_row = {
                    'Bond_Index': idx,
                    'Atom1': bond.GetBeginAtomIdx(),
                    'Atom2': bond.GetEndAtomIdx(),
                    'Bond_Type': str(bond.GetBondType()),
                    'Importance': imp
                }
                if index is not None:
                    bond_data_row['Molecule_Index'] = index
                bond_info.append(bond_data_row)

                highlight_bonds.append(idx)
                bond_colors[idx] = (float(imp_normalized), 0.0, float(1 - imp_normalized))

            bond_importance_df = pd.DataFrame(bond_info)

            try:
                drawer = Draw.MolDraw2DCairo(600, 600)
                op = Draw.MolDrawOptions()
                op.addBondIndices = True
                drawer.SetDrawOptions(op)

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
                st.warning(f"Error generating bond importance map for SMILES {smiles}: {e}")
                bond_img = None
        else:
            st.warning(f"Bond weights array too short for SMILES {smiles}. Expected {num_bonds}, got {len(bond_weights)}.")

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
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# This is placed here because from_smiles needs torch_geometric which is not common across the models
# from_smiles should ideally be imported directly in the page using it.
# However, the processing logic for Data object is very similar.
# Let's create a common function to process the SMILES to a PyG Data object.
from torch_geometric.utils import from_smiles

def smiles_to_graph_data(smiles, device):
    """Converts a SMILES string to a PyG Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.warning(f"Invalid SMILES: {smiles}")
        return None

    try:
        data = from_smiles(smiles).to(device)

        if data.num_nodes == 0 or data.x is None or data.x.nelement() == 0:
            st.warning(f"Empty graph generated for SMILES: {smiles}")
            return None

        data.x = data.x.float()
        # Edge attributes can be empty, so handle that explicitly if models expect it to be None
        if hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.nelement() == 0:
            data.edge_attr = None

        return data
    except Exception as e:
        st.error(f"Error converting SMILES {smiles} to graph: {e}")
        return None
