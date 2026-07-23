#!/usr/bin/env python3
"""
Build DrugCLIP training LMDB from PDBbind raw data.

Produces:
  {output_dir}/train.lmdb   -- training set
  {output_dir}/valid.lmdb   -- validation set
  {output_dir}/dict_mol.txt -- molecule atom-type dictionary
  {output_dir}/dict_pkt.txt -- pocket atom-type dictionary

Each LMDB entry is a pickle-serialised dict:
    smi               : SMILES string
    pocket             : identifier (e.g. "1a4w_A_ASP123")
    atoms              : list of element symbols for the ligand
    coordinates        : list of conformers, each conformer is a list of [x,y,z]
    pocket_atoms       : list of element symbols for pocket residues (excl. H)
    pocket_coordinates : list of [x,y,z] for pocket atoms
    label              : pKd/pKi binding affinity (float, -log10 M); default 1.0

Usage:
    python build_train_lmdb.py \
        --data_root ./data/pdbbind_raw \
        --output_dir ./data/train_data \
        --train_ratio 0.9
"""

import argparse
import logging
import os
import pickle
import re
import sys
import shutil
from collections import defaultdict
from typing import Optional
from multiprocessing import Pool, cpu_count

import lmdb
import numpy as np
from Bio.PDB import PDBParser, Chain, Model, Structure
from Bio.PDB import is_aa
from Bio.PDB.Residue import DisorderedResidue, Residue
from Bio.PDB.Atom import DisorderedAtom
import warnings
from Bio.PDB.StructureBuilder import PDBConstructionWarning
from rdkit import Chem
from tqdm import tqdm

warnings.filterwarnings(action="ignore", category=PDBConstructionWarning)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# parse PDBbind INDEX  (protein-ligand)
# ---------------------------------------------------------------------------
UNITS = {
    "nm": -9, "nmol": -9,
    "um": -6, "umol": -6, "μmol": -6, "μm": -6,
    "mm": -3, "mmol": -3,
    "pm": -12, "pmol": -12,
    "fm": -15, "fmol": -15,
    "m": 0, "mol": 0,
}


def _parse_val(v: str) -> Optional[float]:
    """
    Return the value in Molar.  Examples:  '49uM' -> 49e-6, '0.43uM' -> 4.3e-7.
    Returns None on parse failure.
    """
    v = v.strip()
    m = re.match(r"([\d.]+)\s*(\w*)", v)
    if not m:
        return None
    num = float(m.group(1))
    unit = m.group(2).lower()
    exp = UNITS.get(unit, 0)
    return num * (10 ** exp)


def parse_index(path: str) -> dict:
    """Return {pdb_lower: {'resolution':..., 'binding': label_float, ...}}."""
    recs = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            pdb = parts[0].lower()
            resolution = parts[1]
            year = parts[2]
            binding_str = parts[3]

            label = 1.0  # default
            try:
                if "=" in binding_str:
                    _, val_str = binding_str.split("=", 1)
                    molar = _parse_val(val_str)
                    if molar is not None and molar > 0:
                        label = round(-np.log10(molar), 4)
                elif "<" in binding_str:
                    _, val_str = binding_str.split("<", 1)
                    molar = _parse_val(val_str)
                    if molar is not None and molar > 0:
                        label = round(-np.log10(molar), 4)
            except Exception:
                label = 1.0

            recs[pdb] = {
                "pdb": pdb,
                "resolution": resolution,
                "year": year,
                "binding_str": binding_str,
                "label": label,
            }
    return recs


# ---------------------------------------------------------------------------
# ligand  (mol2 / sdf)
# ---------------------------------------------------------------------------
def read_ligand(mol2_path: str, sdf_path: str) -> Optional[dict]:
    """Read a ligand, return {atoms, coordinates (multi-conformer outer list), smi}."""
    path = mol2_path if os.path.isfile(mol2_path) else sdf_path
    if not os.path.isfile(path):
        return None

    mol = None
    # Try as mol2 first, then as sdf (RDKit automatically detects mol2 if extension is .mol2)
    if path.endswith(".mol2"):
        mol = Chem.MolFromMol2File(path, removeHs=False)
    elif path.endswith(".sdf"):
        suppl = Chem.SDMolSupplier(path, removeHs=False)
        if suppl:
            mol = next(suppl, None)

    if mol is None:
        return None

    # element symbols
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    # 3-D coords  (numpy float32, required by AffinityDataset.astype)
    conf = mol.GetConformer()
    coords = np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
         for i in range(mol.GetNumAtoms())],
        dtype=np.float32,
    )

    # SMILES  (strip hydrogens for canonical SMILES used in tokenisation,
    # but keep H in atom list because the Uni-Mol tokenizer handles them)
    mol_hfree = Chem.RemoveHs(mol)
    smi = Chem.MolToSmiles(mol_hfree, isomericSmiles=True)

    # Coordinates stored as list-of-conformers: outer list length = num confs.
    # Single conformer for PDBbind.
    return {
        "atoms": atoms,
        "coordinates": [coords],
        "smi": smi,
    }


# ---------------------------------------------------------------------------
# pocket  (PDB)
# ---------------------------------------------------------------------------
def _safe_element(atom) -> Optional[str]:
    """Return element string from a Bio.PDB Atom, or None."""
    # Bio.PDB >= 1.79 populates atom.element from the PDB element column (cols 77-78)
    if hasattr(atom, "element") and atom.element and atom.element.strip():
        return atom.element.strip()
    # fallback: guess from atom name (strip digits)
    name = atom.get_name().strip()
    elem = re.sub(r"\d", "", name)
    if len(elem) >= 1 and elem[0].isalpha():
        return elem[0] if len(elem) == 1 else elem[:1]
    return None


def read_pocket_pdb(pocket_path: str) -> Optional[dict]:
    """Read a pre-extracted pocket.pdb (all atoms are already the binding site)."""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("pkt", pocket_path)
        model = structure[0]
    except Exception:
        return None

    atoms = []
    coords = []
    for chain in model:
        for res in chain:
            for atom in res:
                elem = _safe_element(atom)
                if elem is None or elem == "H":
                    continue
                atoms.append(elem)
                coords.append(list(atom.get_coord()))

    if not atoms:
        return None
    return {"atoms": atoms, "coordinates": coords}


def extract_pocket_from_protein(
    protein_path: str, lig_coords: list, dist_threshold: float = 6.0
) -> Optional[dict]:
    """Extract pocket atoms from a full protein PDB using ligand coords.

    Returns residues whose any heavy-atom is within *dist_threshold* of any
    ligand heavy atom.
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("prot", protein_path)
        model = structure[0]
    except Exception:
        return None

    lig_array = np.array(lig_coords, dtype=np.float32)

    pocket_atoms = []
    pocket_coords = []

    for chain in model:
        for res in chain:
            if not is_aa(res, standard=True):
                continue
            heavy_res_coords = []
            heavy_res_atoms = []
            for atom in res:
                elem = _safe_element(atom)
                if elem is None or elem == "H":
                    continue
                heavy_res_atoms.append(atom)
                heavy_res_coords.append(atom.get_coord())
            if not heavy_res_coords:
                continue

            res_array = np.array(heavy_res_coords, dtype=np.float32)
            # minimum distance between any pair of atoms
            dist = np.linalg.norm(
                res_array[:, None, :] - lig_array[None, :, :], axis=-1
            ).min()
            if dist <= dist_threshold:
                for atom in heavy_res_atoms:
                    elem = _safe_element(atom)
                    if elem:
                        pocket_atoms.append(elem)
                        pocket_coords.append(list(atom.get_coord()))

    if not pocket_atoms:
        return None
    return {"atoms": pocket_atoms, "coordinates": pocket_coords}


# ---------------------------------------------------------------------------
# LMDB helpers
# ---------------------------------------------------------------------------
MAP_SIZE = 20 * 1024 ** 3  # 20 GB


def _open_lmdb_write(path: str):
    return lmdb.open(
        path,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        map_size=MAP_SIZE,
    )


def write_lmdb(entries: list, path: str):
    """entries: list of dicts."""
    env = _open_lmdb_write(path)
    with env.begin(write=True) as txn:
        for idx, entry in enumerate(entries):
            txn.put(str(idx).encode("ascii"), pickle.dumps(entry))
    env.close()


# ---------------------------------------------------------------------------
# dictionary files  (will be placed alongside train.lmdb / valid.lmdb)
# ---------------------------------------------------------------------------
MOL_DICT = [
    "[PAD]", "[CLS]", "[SEP]", "[UNK]",
    "C", "N", "O", "S", "H", "Cl", "F", "Br", "I", "Si", "P", "B",
    "Na", "K", "Al", "Ca", "Sn", "As", "Hg", "Fe", "Zn", "Cr", "Se",
    "Gd", "Au", "Li",
]

PKT_DICT = [
    "[PAD]", "[CLS]", "[SEP]", "[UNK]",
    "C", "N", "O", "S", "H",
]


def write_dict_txt(path: str, items: list):
    with open(path, "w") as f:
        for item in items:
            f.write(item + "\n")


# ---------------------------------------------------------------------------
# per-complex worker  (module-level for pickling)
# ---------------------------------------------------------------------------
def _process_one(comp: dict):
    """Process a single PDBbind complex.  Returns the LMDB entry dict or None."""
    lig = read_ligand(comp["mol2"], comp["sdf"])
    if lig is None:
        return None

    pocket_data = read_pocket_pdb(comp["pocket"])
    if pocket_data is None:
        if os.path.isfile(comp["protein"]):
            pocket_data = extract_pocket_from_protein(
                comp["protein"], lig["coordinates"][0]
            )
    if pocket_data is None:
        return None

    return {
        "smi": lig["smi"],
        "pocket": comp["pdb"],
        "atoms": lig["atoms"],
        "coordinates": lig["coordinates"],
        "pocket_atoms": pocket_data["atoms"],
        "pocket_coordinates": pocket_data["coordinates"],
        "label": comp["label"],
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build DrugCLIP training LMDB from PDBbind")
    parser.add_argument(
        "--data_root", type=str, default="./data/pdbbind_raw",
        help="root directory of PDBbind raw data (contains index/ and P-L/)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data/pdbbind_train",
        help="output directory for LMDB files and dictionaries"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.9,
        help="fraction of data for training set"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="random seed for splitting"
    )
    parser.add_argument(
        "--workers", type=int, default=cpu_count(),
        help="number of parallel workers (default: all cores)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 1. parse INDEX ----------------------------------------------------
    index_files = [
        f for f in os.listdir(os.path.join(args.data_root, "index"))
        if f.startswith("INDEX_general_PL") and f.endswith(".lst")
    ]
    if not index_files:
        logger.error("No INDEX_general_PL.*.lst found in index/ directory")
        sys.exit(1)
    index_path = os.path.join(args.data_root, "index", index_files[0])
    logger.info(f"Parsing INDEX: {index_path}")
    index_data = parse_index(index_path)
    logger.info(f"  {len(index_data)} entries parsed")

    # ---- 2. scan complex directories ---------------------------------------
    pl_dir = os.path.join(args.data_root, "P-L")
    all_complexes = []
    for subdir in sorted(os.listdir(pl_dir)):
        subpath = os.path.join(pl_dir, subdir)
        if not os.path.isdir(subpath):
            continue
        for pdb_name in sorted(os.listdir(subpath)):
            pdb_path = os.path.join(subpath, pdb_name)
            if not os.path.isdir(pdb_path):
                continue
            mol2 = os.path.join(pdb_path, f"{pdb_name}_ligand.mol2")
            sdf = os.path.join(pdb_path, f"{pdb_name}_ligand.sdf")
            pocket = os.path.join(pdb_path, f"{pdb_name}_pocket.pdb")
            protein = os.path.join(pdb_path, f"{pdb_name}_protein.pdb")
            all_complexes.append({
                "pdb": pdb_name.lower(),
                "dir": pdb_path,
                "mol2": mol2,
                "sdf": sdf,
                "pocket": pocket,
                "protein": protein,
            })

    logger.info(f"  {len(all_complexes)} complex directories found")

    # ---- 3. pre-attach labels from INDEX ------------------------------------
    for comp in all_complexes:
        idx_rec = index_data.get(comp["pdb"], {})
        comp["label"] = idx_rec.get("label", 1.0)

    # ---- 4. process complexes in parallel -----------------------------------
    logger.info(f"Processing with {args.workers} workers ...")
    with Pool(processes=args.workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(_process_one, all_complexes, chunksize=100),
            total=len(all_complexes),
            desc="Processing complexes",
        ))

    entries = [r for r in results if r is not None]
    n_skipped = len(all_complexes) - len(entries)

    logger.info(f"  {len(entries)} complexes successfully processed")
    logger.info(f"  Skipped: {n_skipped}")

    if not entries:
        logger.error("No valid entries generated. Exiting.")
        sys.exit(1)

    # ---- 5. train / valid split --------------------------------------------
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(entries))
    n_train = int(len(entries) * args.train_ratio)
    train_idx = indices[:n_train]
    valid_idx = indices[n_train:]

    train_entries = [entries[i] for i in train_idx]
    valid_entries = [entries[i] for i in valid_idx]
    logger.info(f"  Train: {len(train_entries)},  Valid: {len(valid_entries)}")

    # ---- 6. write LMDB ------------------------------------------------------
    train_path = os.path.join(args.output_dir, "train.lmdb")
    valid_path = os.path.join(args.output_dir, "valid.lmdb")

    logger.info(f"Writing {train_path} ...")
    write_lmdb(train_entries, train_path)

    logger.info(f"Writing {valid_path} ...")
    write_lmdb(valid_entries, valid_path)

    # ---- 7. copy / write dictionary files -----------------------------------
    # Use the existing dict files in ./dict/ if available, otherwise write new ones
    src_dict_mol = os.path.join(os.path.dirname(__file__), "dict", "dict_mol.txt")
    src_dict_pkt = os.path.join(os.path.dirname(__file__), "dict", "dict_pkt.txt")

    dst_dict_mol = os.path.join(args.output_dir, "dict_mol.txt")
    dst_dict_pkt = os.path.join(args.output_dir, "dict_pkt.txt")

    if os.path.isfile(src_dict_mol):
        shutil.copy2(src_dict_mol, dst_dict_mol)
        logger.info(f"Copied dict_mol.txt from {src_dict_mol}")
    else:
        write_dict_txt(dst_dict_mol, MOL_DICT)
        logger.info("Wrote default dict_mol.txt")

    if os.path.isfile(src_dict_pkt):
        shutil.copy2(src_dict_pkt, dst_dict_pkt)
        logger.info(f"Copied dict_pkt.txt from {src_dict_pkt}")
    else:
        write_dict_txt(dst_dict_pkt, PKT_DICT)
        logger.info("Wrote default dict_pkt.txt")

    logger.info("Done.")


if __name__ == "__main__":
    main()
