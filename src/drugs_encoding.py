#!/usr/bin/env python3

from rdkit.Chem import AllChem, rdqueries
from rdkit import Chem
import requests
import numpy as np
import pandas as pd


def get_canonical_smiles_by_name(drug_name):
    API_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/JSON"
    resp = requests.get(API_URL.format(name=drug_name))
    rd = resp.json()
    if 'PC_Compounds' not in rd:
        return None

    smiles = str(rd['PC_Compounds'][0]['props'][18]['value']['sval'])
    return smiles

def get_canonical_smiles_by_id(drug_id):
    API_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{drug_id}/JSON"
    resp = requests.get(API_URL.format(drug_id=drug_id))
    rd = resp.json()
    if 'PC_Compounds' not in rd:
        return None

    props = rd['PC_Compounds'][0]['props']
    for prop in props:
        if prop['urn']['label'] == 'SMILES' and prop['urn']['name'] == 'Canonical':
            return prop['value']['sval']

    return None

def get_morgan_fingerprint(smiles_repr, n_bits=256):
    mol = Chem.MolFromSmiles(smiles_repr)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=n_bits)

    out = fp.ToList()

    return out


def discretize_values(val, val_list, unknown_k='Unknown'):
    s_list = sorted(val_list)

    if unknown_k not in s_list:
        s_list = s_list + [unknown_k]

    if val not in s_list:
        val = unknown_k

    idx = s_list.index(val)
    out = np.zeros(len(s_list), dtype=np.int32)
    out[idx] = 1

    return out.tolist()


def get_atomic_features(smiles_repr):
    '''
    Given a smiles representation of the molecule it returns atom's specific features


    0. Atomic Symbol :
    1. xAtomic number : number of protons found in the nucleus
    2. Chirality: superimposable ??
    3. xDegree of atom : number of bonded atoms including Hs
    4. xFrmal charge
    5. xTotal number of H of the atom
    6. xNumber of radical electrons(?)
    7. Hybridization of the atom
    8. xAromatic atom?
    9. xAtom is in ring or not?

    '''

    mol = Chem.MolFromSmiles(smiles_repr)
    if mol is None:
        return None,None,None
    out_features = []
    q = rdqueries.IsAromaticQueryAtom()
    aromatic_atoms = ([x.GetIdx() for x in mol.GetAtomsMatchingQuery(q)])

    bonds = []
    edge_attrs = []

    for b in mol.GetBonds():
        bonds.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        edge_attrs.append((b.GetBondTypeAsDouble(), int(b.IsInRing())))

    #print(b.GetBeginAtomIdx(),b.GetEndAtomIdx(),
    #      b.GetBondType(),b.GetStereo())



    for i, atom in enumerate(mol.GetAtoms()):
        atomic_symbol = discretize_values(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br',
                                           'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V',
                                           'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd',
                                           'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown'])

        atomic_number = atom.GetAtomicNum()
        atomic_degree = len(atom.GetNeighbors())
        atomic_formal_charge = atom.GetFormalCharge()
        atom_inring = int(atom.IsInRing())
        atom_aromatic = int(i in aromatic_atoms)
        atom_radical = int(atom.GetNumRadicalElectrons())
        atom_total_hs = int(atom.GetTotalNumHs())

        out_features.append(atomic_symbol + [
            atomic_number,
            atomic_degree,
            atomic_formal_charge,
            atom_inring,
            atom_radical,
            atom_total_hs,
            atom_aromatic])

    return out_features, bonds, edge_attrs


def get_smiles_drugbank(drugbank_id):
    headers = {"Accept": "application/json"}


    response = requests.get(f"https://go.drugbank.com/drugs/{drugbank_id}.smiles", headers=headers)

    print(f"https://go.drugbank.com/drugs/{drugbank_id}.smiles")
    if response.status_code == 200:
        smiles = response.text.strip()
        print(f"The SMILES code for DrugBank ID {drugbank_id} is: {smiles}")
        return smiles
    else:
        print(f"Failed, status code: {response.status_code}")
        return None




if __name__ == '__main__':
    pass
