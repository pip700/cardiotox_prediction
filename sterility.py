import pandas as pd
from rdkit import Chem
import sys
import os

def remove_stereochemistry(smiles):
    """Remove stereochemistry from a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Could not parse SMILES: {smiles}")
            return None

        # Remove stereo chemistry
        Chem.RemoveStereochemistry(mol)

        # Convert back to SMILES
        clean_smiles = Chem.MolToSmiles(mol)
        return clean_smiles

    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {e}")
        return None


def process_csv(input_file, output_file, smiles_column="SMILES"):
    """Process a CSV file and remove stereochemistry from SMILES."""

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    # Read CSV
    print(f"Reading file: {input_file}")
    df = pd.read_csv(input_file)

    # Check if SMILES column exists
    if smiles_column not in df.columns:
        print(f"Error: Column '{smiles_column}' not found in CSV.")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    print(f"Processing {len(df)} molecules...")

    # Apply stereochemistry removal
    df[smiles_column] = df[smiles_column].apply(
        lambda x: remove_stereochemistry(x) if pd.notna(x) else None
    )

    # Save output CSV
    df.to_csv(output_file, index=False)
    print(f"Done! Output saved to: {output_file}")
    print(f"Total molecules processed: {len(df)}")
    print(f"Failed conversions: {df[smiles_column].isna().sum()}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # --- Edit these variables or pass arguments via CLI ---
    INPUT_FILE    = "input.csv"
    OUTPUT_FILE   = "output_no_stereo.csv"
    SMILES_COLUMN = "SMILES"          # change to your column name

    # Optional: allow CLI arguments  →  script.py input.csv output.csv SMILES
    if len(sys.argv) == 4:
        INPUT_FILE, OUTPUT_FILE, SMILES_COLUMN = sys.argv[1], sys.argv[2], sys.argv[3]
    elif len(sys.argv) == 3:
        INPUT_FILE, OUTPUT_FILE = sys.argv[1], sys.argv[2]
    elif len(sys.argv) == 2:
        INPUT_FILE = sys.argv[1]

    process_csv(INPUT_FILE, OUTPUT_FILE, SMILES_COLUMN)
