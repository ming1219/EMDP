"""
De Novo 含能分子数据准备脚本
通过允许的token列表过滤SMILES

用法:
    python energetic_mol_design/scripts/prepare_denovo_data.py --input smiles.csv
"""

import pandas as pd
from rdkit import Chem
import os
import argparse
import random
import re
from typing import List, Set
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(BASE_DIR, "smiles.csv")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "data")

REINVENT_ALLOWED_TOKENS = {
    '#', '[nH]', 'o', '(', 'Br', '[O-]', '7', '^', '[N-]', '6', '1', '9', 
    '[S+]', 'Cl', '2', '$', ')', '-', '=', '4', 'N', 's', 'n', '3', 'c', 
    'O', '[n+]', '5', '%10', 'C', '8', 'S', 'F', '[N+]'
}


def tokenize_smiles(smiles: str) -> List[str]:
    """将SMILES字符串分解为token列表"""
    pattern = r"(\[[^\]]+\]|Br|Cl|%\d{2}|.)"
    return re.findall(pattern, smiles)


def validate_reinvent_smiles(smiles: str, allowed_tokens: Set[str]) -> bool:
    """验证SMILES是否只包含允许的token"""
    if not smiles:
        return False
    tokens = tokenize_smiles(smiles)
    for token in tokens:
        if token not in allowed_tokens:
            return False
    return True


def prepare_denovo_data(smiles_list: List[str], allowed_tokens: Set[str]) -> List[str]:
    """准备De Novo训练数据，通过允许的token过滤"""
    results = []
    
    for smi in tqdm(smiles_list, desc="验证SMILES token"):
        if not smi or not isinstance(smi, str):
            continue
        smi = smi.strip()
        
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        
        canonical = Chem.MolToSmiles(mol, canonical=True)
        
        if validate_reinvent_smiles(canonical, allowed_tokens):
            results.append(canonical)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='准备De Novo训练数据')
    parser.add_argument('--input', '-i', default=DEFAULT_INPUT, help='输入SMILES文件')
    parser.add_argument('--output-dir', '-o', default=DEFAULT_OUTPUT_DIR, help='输出目录')
    parser.add_argument('--valid-ratio', '-v', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--seed', '-s', type=int, default=42, help='随机种子')
    parser.add_argument('--smiles-col', type=str, default=None, help='SMILES列名')
    
    args = parser.parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"读取: {args.input}")
    df = pd.read_csv(args.input)
    smiles_col = args.smiles_col if args.smiles_col else df.columns[0]
    smiles_list = df[smiles_col].dropna().tolist()
    print(f"原始分子数: {len(smiles_list)}")
    
    filtered_smiles = prepare_denovo_data(smiles_list, REINVENT_ALLOWED_TOKENS)
    filtered_smiles = list(set(filtered_smiles))
    random.shuffle(filtered_smiles)
    print(f"过滤后分子数: {len(filtered_smiles)}")
    
    valid_size = int(len(filtered_smiles) * args.valid_ratio)
    train_smiles = filtered_smiles[valid_size:]
    valid_smiles = filtered_smiles[:valid_size]
    
    train_file = os.path.join(args.output_dir, 'denovo_training_data.smi')
    with open(train_file, 'w') as f:
        for smi in train_smiles:
            f.write(f"{smi}\n")
    
    valid_file = os.path.join(args.output_dir, 'denovo_validation_data.smi')
    with open(valid_file, 'w') as f:
        for smi in valid_smiles:
            f.write(f"{smi}\n")
    
    print(f"训练集: {len(train_smiles)}, 验证集: {len(valid_smiles)}")
    print(f"输出: {args.output_dir}")


if __name__ == '__main__':
    main()
