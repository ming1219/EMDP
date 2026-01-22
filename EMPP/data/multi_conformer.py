from __future__ import absolute_import, division, print_function

import os
import warnings

import numpy as np
import time
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import rdMolAlign
from rdkit.ML.Cluster import Butina
from scipy.spatial.distance import pdist

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings(action='ignore')
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from ..config import MODEL_CONFIG
from ..utils import logger
from ..weights import WEIGHT_DIR, weight_download
from .dictionary import Dictionary

from .conformer import coords2unimol, inner_smi2coords


class MultiConformerGen(object):
    def __init__(self, **params):
        self._init_features(**params)

    def _init_features(self, **params):
        self.seed = params.get('seed', 42)
        self.max_atoms = params.get('max_atoms', 256)
        self.data_type = params.get('data_type', 'molecule')
        self.remove_hs = params.get('remove_hs', False)
        self.mmff_optimize = params.get('mmff_optimize', True)
        self.is_train = params.get('is_train', True)
        self.topk_k = params.get('topk_k', 3)

        self.method = params.get('method', 'rdkit_etkdg')
        self.num_conformers = params.get('num_conformers', 1)
        self.max_clusters = params.get('max_clusters', None)

        if self.data_type == 'molecule':
            name = "no_h" if self.remove_hs else "all_h"
            name = self.data_type + '_' + name
            self.dict_name = MODEL_CONFIG['dict'][name]
        else:
            self.dict_name = MODEL_CONFIG['dict'][self.data_type]
        
        if not os.path.exists(os.path.join(WEIGHT_DIR, self.dict_name)):
            weight_download(self.dict_name, WEIGHT_DIR)
        self.dictionary = Dictionary.load(os.path.join(WEIGHT_DIR, self.dict_name))
        self.dictionary.add_symbol("[MASK]", is_special=True)

        self.multi_process = params.get('multi_process', True)

    def transform_mols(self, mols_list):
        inputs = []
        for mol in mols_list:
            if mol.GetNumConformers() == 0:
                try:
                    AllChem.EmbedMolecule(mol, useRandomCoords=True)
                    try:
                        AllChem.MMFFOptimizeMolecule(mol)
                    except Exception:
                        pass
                except Exception:
                    pass
            atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
            coordinates = mol.GetConformer().GetPositions().astype(np.float32)
            feat = coords2unimol(
                atoms,
                coordinates,
                self.dictionary,
                self.max_atoms,
                remove_hs=self.remove_hs,
            )
            inputs.append(feat)
        return inputs

    def single_process(self, smiles):
        try:
            mol, coords_energy_list = multi_conformer_generation(
                smiles,
                seed=self.seed,
                num_conformers=self.num_conformers,
                mmff_optimize=self.mmff_optimize,
                max_clusters=self.max_clusters,
                topk_k=self.topk_k,
            )
            if mol is None or coords_energy_list is None:
                logger.warning(f"Using placeholder for failed molecule: {smiles}")
                return self._get_placeholder_result()
            atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
            feats = []
            for coords, energy in coords_energy_list:
                feat = coords2unimol(
                    atoms,
                    coords,
                    self.dictionary,
                    self.max_atoms,
                    remove_hs=self.remove_hs,
                    energy=energy
                )
                feats.append(feat)
            return feats, mol, False
        except Exception as e:
            logger.warning(f"Using placeholder for failed molecule: {smiles}, error: {str(e)}")
            return self._get_placeholder_result()

    def _get_placeholder_result(self):
        """生成一个固定的占位构象（甲烷），用于失败的分子"""
        try:
            mol = Chem.MolFromSmiles('C')
            mol = AllChem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=self.seed)
            coords = mol.GetConformer().GetPositions().astype(np.float32)
            atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
            feats = []
            for _ in range(self.topk_k):
                feat = coords2unimol(
                    atoms,
                    coords,
                    self.dictionary,
                    self.max_atoms,
                    remove_hs=self.remove_hs,
                    energy=0.0
                )
                feats.append(feat)
            return feats, mol, True
        except Exception as e:
            logger.error(f"Failed to create placeholder molecule: {str(e)}")
            return None

    def transform_raw(self, atoms_list, coordinates_list):
        inputs = []
        for atoms, coordinates in zip(atoms_list, coordinates_list):
            inputs.append(
                coords2unimol(
                    atoms,
                    coordinates,
                    self.dictionary,
                    self.max_atoms,
                    remove_hs=self.remove_hs,
                )
            )
        return inputs

    def transform(self, smiles_list):
        if self.multi_process:
            workers = min(8, os.cpu_count() or 1)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(tqdm(executor.map(self.single_process, smiles_list), total=len(smiles_list)))
        else:
            results = [self.single_process(smiles) for smiles in tqdm(smiles_list)]

        placeholder_count = sum(1 for r in results if r is not None and len(r) == 3 and r[2])
        failed_count = sum(1 for r in results if r is None)
        
        if placeholder_count > 0:
            logger.warning(f"Used placeholder for {placeholder_count}/{len(results)} molecules due to conformer generation failure")
        if failed_count > 0:
            logger.error(f"Completely failed {failed_count}/{len(results)} molecules (even placeholder failed)")

        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            logger.error("All molecules failed to generate conformers")
            return [], []
        
        inputs = [r[0] for r in valid_results]
        mols = [r[1] for r in valid_results]
        return inputs, mols

def multi_conformer_generation(smi, seed=42, num_conformers=1, mmff_optimize=True,
                             mmff_max_iters=200, max_clusters=None, topk_k=3):
    """
    生成多构象并返回模型需要的数据。
    :param smi: SMILES 字符串
    :param seed: 随机种子
    :param num_conformers: 生成的构象数量
    :param mmff_optimize: 是否使用 MMFF 优化
    :param mmff_max_iters: MMFF 优化最大迭代次数
    :param max_clusters: 最大聚类数
    :param topk_k: 返回的 top-k 构象数量
    :return: (mol, coords_energy_list) - mol 是模板分子，coords_energy_list 是 [(coords, energy), ...] 列表
    """
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f'Invalid SMILES: {smi}')
        mol = AllChem.AddHs(mol)
        if mol.GetNumAtoms() == 0:
            raise ValueError(f'No atoms in molecule: {smi}')
        conformers_data = generate_multiple_conformers(
            mol, seed, num_conformers, mmff_optimize=mmff_optimize,
            mmff_max_iters=mmff_max_iters
        )
        if not conformers_data:
            return fallback_single_conformer(smi, seed, topk_k=topk_k)

        filtered_conformers = energy_prefilter_3kcal(conformers_data)

        representatives, _ = cluster_and_select_representatives(
            filtered_conformers, cutoff=0.8, max_clusters=max_clusters
        )

        score_conformers_for_energetic_materials(representatives)
        representatives.sort(key=lambda x: x.get('total_score', -x['energy']), reverse=True)

        best_conformers = representatives[:topk_k]

        for conf in best_conformers:
            optimize_conformer_with_xtb(conf)

        while len(best_conformers) < topk_k and best_conformers:
            best_conformers.append({
                'mol': Chem.Mol(best_conformers[0]['mol']),
                'coordinates': best_conformers[0]['coordinates'].copy(),
                'energy': best_conformers[0]['energy']
            })

        coords_energy_list = [
            (np.array(conf['coordinates'], dtype=np.float32), conf['energy'])
            for conf in best_conformers
        ]
        
        mol_result = best_conformers[0]['mol'] if best_conformers else mol
        return mol_result, coords_energy_list
        
    except Exception as e:
        logger.error(f"Failed to generate multi-conformers for {smi}: {str(e)}")
        return fallback_single_conformer(smi, seed, topk_k=topk_k)

def generate_multiple_conformers(mol, seed, num_conformers, mmff_optimize=True, mmff_max_iters=200):
    conformers_data = []
    try:
        if mol is None:
            logger.error("Input molecule is None")
            return []
        if mol.GetNumAtoms() == 0:
            logger.error("Molecule has no atoms")
            return []
        logger.info(f"开始构象嵌入: 目标 {num_conformers}")
        mol_with_hs = AllChem.AddHs(mol)
        all_conformers = []
        max_attempts = 5
        attempts = 0
        embed_start = time.time()
        while len(all_conformers) < num_conformers and attempts < max_attempts:
            attempts += 1
            temp_mol = Chem.Mol(mol_with_hs)
            try:
                params = AllChem.ETKDGv3()
                params.randomSeed = seed + attempts
                params.numThreads = 0
                params.enforceChirality = True
                params.useRandomCoords = True
                needed = num_conformers - len(all_conformers)
                if needed <= 0:
                    break
                new_conf_ids = AllChem.EmbedMultipleConfs(
                    temp_mol,
                    numConfs=needed,
                    params=params
                )
                if new_conf_ids:
                    for conf_id in new_conf_ids:
                        conf = temp_mol.GetConformer(conf_id)
                        new_conf_id = mol_with_hs.AddConformer(conf, assignId=True)
                        all_conformers.append(new_conf_id)
                    if len(all_conformers) >= num_conformers:
                        break
            except Exception as e:
                logger.debug(f"ETKDGv3 attempt {attempts} failed: {str(e)}")
        logger.info(f"嵌入完成: {len(all_conformers)} 构象, 用时 {time.time() - embed_start:.2f}s")
        if len(all_conformers) == 0:
            logger.warning("ETKDGv3 completely failed, trying fallback method")
            try:
                fallback_start = time.time()
                conf_ids = AllChem.EmbedMultipleConfs(
                    mol_with_hs,
                    numConfs=num_conformers,
                    randomSeed=seed,
                    clearConfs=True
                )
                logger.info(f"备用嵌入生成 {len(conf_ids)} 构象, 用时 {time.time() - fallback_start:.2f}s")
                all_conformers = list(conf_ids)
            except Exception as e2:
                logger.error(f"All conformer generation methods failed: {str(e2)}")
                return []
        if len(all_conformers) == 0:
            logger.warning("No conformers generated")
            return []
        if mmff_optimize:
            try:
                logger.info(f"批量MMFF优化开始: {len(all_conformers)} 构象")
                batch_opt_start = time.time()
                AllChem.MMFFOptimizeMoleculeConfs(mol_with_hs, maxIters=int(mmff_max_iters))
                logger.info(f"批量MMFF优化完成, 用时 {time.time() - batch_opt_start:.2f}s")
            except Exception as e:
                logger.warning(f"Batch MMFF optimization failed: {str(e)}")
        
        pbar_conf = tqdm(total=len(all_conformers), desc="构象能量计算", leave=False)
        for i, conf_id in enumerate(all_conformers):
            try:
                single_mol = Chem.Mol(mol_with_hs)
                single_mol.RemoveAllConformers()
                single_mol.AddConformer(mol_with_hs.GetConformer(conf_id), assignId=True)
                energy = None
                converged = True
                try:
                    mp = AllChem.MMFFGetMoleculeProperties(single_mol)
                    if mp:
                        ff = AllChem.MMFFGetMoleculeForceField(single_mol, mp)
                        if ff is not None:
                            energy = ff.CalcEnergy()
                        else:
                            ff = AllChem.UFFGetMoleculeForceField(single_mol)
                            if ff is not None:
                                energy = ff.CalcEnergy()
                            else:
                                energy = 0.0
                    else:
                        ff = AllChem.UFFGetMoleculeForceField(single_mol)
                        if ff is not None:
                            energy = ff.CalcEnergy()
                        else:
                            energy = 0.0
                except Exception as e:
                    energy = 0.0
                coordinates = single_mol.GetConformer().GetPositions().astype(np.float32)
                conformers_data.append({
                    'mol': single_mol,
                    'coordinates': coordinates,
                    'conf_id': i,
                    'energy': energy,
                    'converged': converged
                })
                pbar_conf.update(1)
            except Exception as e:
                continue
        pbar_conf.close()
        logger.info(f"构象优化与能量计算完成: {len(conformers_data)} 构象")
        return conformers_data
    except Exception as e:
        logger.error(f"Conformer generation failed: {str(e)}")
        return []


def cluster_and_select_representatives(conformers_data, cutoff=0.8, max_clusters=None):
    if not conformers_data:
        return [], {'reason': 'no_conformer', 'cluster_count': 0, 'representative_count': 0}
    heavy_mols = [Chem.RemoveHs(conf['mol']) for conf in conformers_data]
    n = len(heavy_mols)
    dists = []
    D = [[0.0] * n for _ in range(n)]
    total_pairs = n * (n - 1) // 2
    pbar = tqdm(total=total_pairs, desc="RMSD筛选", leave=False)
    for i in range(n):
        for j in range(i + 1, n):
            rms = rdMolAlign.GetBestRMS(heavy_mols[i], heavy_mols[j])
            D[i][j] = rms
            D[j][i] = rms
            dists.append(rms)
            pbar.update(1)
    pbar.close()
    clusters = Butina.ClusterData(dists, n, isDistData=True, distThresh=cutoff)
    original_cluster_count = len(clusters)
    used_cutoff = cutoff
    if max_clusters is not None and original_cluster_count > max_clusters:
        step = 0.1
        limit = cutoff + 2.0
        while len(clusters) > max_clusters and used_cutoff < limit:
            used_cutoff += step
            clusters = Butina.ClusterData(dists, n, isDistData=True, distThresh=used_cutoff)
        if len(clusters) > max_clusters:
            clusters = sorted(clusters, key=lambda cl: len(cl), reverse=True)[:max_clusters]
    reps = []
    for clus in clusters:
        idxs = list(clus)
        sums = [(k, sum(D[k][m] for m in idxs if m != k)) for k in idxs]
        rep_idx = min(sums, key=lambda x: x[1])[0]
        reps.append(conformers_data[rep_idx])
    info = {
        'reason': 'rmsd_cluster_medoid',
        'cluster_count': len(clusters),
        'representative_count': len(reps),
        'cutoff': used_cutoff,
        'original_cluster_count': original_cluster_count,
        'max_clusters': max_clusters if max_clusters is not None else -1,
    }
    return reps, info

def cluster_and_select_representatives_usrcat(conformers_data, cutoff=4.0, max_clusters=None):
    if not conformers_data:
        logger.warning("USRCAT聚类: 输入构象列表为空")
        return [], {'reason': 'no_conformer', 'cluster_count': 0, 'representative_count': 0}
    
    n = len(conformers_data)
    logger.info(f"USRCAT聚类开始: 输入 {n} 个构象, cutoff={cutoff}")

    fps = []
    for conf in conformers_data:
        fp = rdMolDescriptors.GetUSRCAT(conf['mol'])
        fps.append(fp)
    logger.info(f"USRCAT指纹计算完成: {len(fps)}个, 每个{len(fps[0])}维")

    dists = []
    for i in range(1, n):
        for j in range(i):
            dist = sum(abs(a - b) for a, b in zip(fps[i], fps[j]))
            dists.append(dist)
    
    if dists:
        logger.info(f"距离矩阵: min={min(dists):.2f}, max={max(dists):.2f}, mean={sum(dists)/len(dists):.2f}")

    clusters = Butina.ClusterData(dists, n, isDistData=True, distThresh=cutoff)
    original_cluster_count = len(clusters)
    logger.info(f"Butina聚类: {original_cluster_count} 个簇 (cutoff={cutoff})")
    
    used_cutoff = cutoff
    if max_clusters is not None and original_cluster_count > max_clusters:
        logger.info(f"簇数超限, 调整cutoff...")
        step = 0.5
        limit = cutoff + 10.0
        while len(clusters) > max_clusters and used_cutoff < limit:
            used_cutoff += step
            clusters = Butina.ClusterData(dists, n, isDistData=True, distThresh=used_cutoff)
        
        if len(clusters) > max_clusters:
            clusters = sorted(clusters, key=lambda cl: len(cl), reverse=True)[:max_clusters]

    reps = []
    for i, clus in enumerate(clusters):
        best_idx = min(clus, key=lambda idx: conformers_data[idx]['energy'])
        best_energy = conformers_data[best_idx]['energy']
        reps.append(conformers_data[best_idx])
        logger.debug(f"  簇{i+1}: {len(clus)}个构象 → 选择idx={best_idx}, E={best_energy:.2f} kcal/mol")

    logger.info(f"USRCAT聚类完成: {len(reps)}个代表构象")
    
    info = {
        'reason': 'usrcat_cluster_min_energy',
        'cluster_count': len(clusters),
        'representative_count': len(reps),
        'cutoff': used_cutoff,
        'original_cluster_count': original_cluster_count,
        'max_clusters': max_clusters if max_clusters is not None else -1,
    }
    return reps, info

def optimize_conformer_with_xtb(conformer):
    """
    使用 XTB (GFN2-xTB) 优化单个构象
    
    能量单位说明：
    - XTB 输出能量单位是 eV (电子伏特)
    - 转换为 kcal/mol: 1 eV = 23.0605 kcal/mol
    - 最终所有能量统一为 kcal/mol
    """
    try:
        from xtb.ase.calculator import XTB
        from ase import Atoms
        from ase.optimize import BFGS
        import numpy as np
        
        logger.info(f"开始 XTB (GFN2-xTB) 优化构象 {conformer.get('conf_id', 'unknown')}...")

        rd_mol = conformer['mol']
        pos = conformer['coordinates']
        symbols = [a.GetSymbol() for a in rd_mol.GetAtoms()]
        
        ase_atoms = Atoms(symbols=symbols, positions=pos)

        calc = XTB(method="GFN2-xTB", accuracy=1.0, max_iterations=250)
        ase_atoms.calc = calc

        opt = BFGS(ase_atoms, logfile=None)
        opt.run(fmax=0.05) 

        new_pos = ase_atoms.get_positions()
        potential_energy_eV = ase_atoms.get_potential_energy()

        conf = rd_mol.GetConformer()
        for i in range(len(symbols)):
            x, y, z = float(new_pos[i][0]), float(new_pos[i][1]), float(new_pos[i][2])
            conf.SetAtomPosition(i, (x, y, z))

        EV_TO_KCAL = 23.0605
        conformer['coordinates'] = new_pos.astype(np.float32)
        conformer['energy'] = potential_energy_eV * EV_TO_KCAL
        conformer['xtb_optimized'] = True
        conformer['energy_unit'] = 'kcal/mol'
        
        logger.info(f"XTB 优化完成。能量: {conformer['energy']:.4f} kcal/mol")
        return True
    except ImportError:
        logger.warning("XTB or ASE not installed, skipping XTB optimization")
        conformer['xtb_optimized'] = False
        conformer['energy_unit'] = 'kcal/mol'
        return False
    except Exception as e:
        logger.warning(f"XTB optimization failed: {str(e)}, using MMFF coordinates")
        conformer['xtb_optimized'] = False
        conformer['energy_unit'] = 'kcal/mol'
        return False

def score_conformers_for_energetic_materials(conformers_data):

    energies = [conf['energy'] for conf in conformers_data if conf['energy'] is not None]
    
    if not energies:
        for conf in conformers_data:
            conf['total_score'] = 0.0
            conf['energy_score'] = 0.0
            conf['compactness_score'] = 0.0
        return

    min_energy = min(energies)
    max_energy = max(energies)

    for conf in conformers_data:
        mol = conf['mol']
        coordinates = conf['coordinates']

        if conf['energy'] is not None and max_energy > min_energy:
            conf['energy_score'] = (max_energy - conf['energy']) / (max_energy - min_energy)
        else:
            conf['energy_score'] = 1.0

        conf['compactness_score'] = calculate_compactness_score(mol, coordinates)

        conf['total_score'] = (
            0.50 * conf['energy_score'] +
            0.50 * conf['compactness_score']
        )

def select_for_energetic_materials(conformers_data):

    print("-" * 60)
    print("构象选择方法: 50%能量 + 50%紧凑性综合评分")
    print(f"开始构象选择，总共有 {len(conformers_data)} 个构象")
    
    if len(conformers_data) == 1:
        print("只有1个构象，直接选择")
        return conformers_data[0]

    energies = [conf['energy'] for conf in conformers_data if conf['energy'] is not None]
    energy_none_count = len([conf for conf in conformers_data if conf['energy'] is None])
    
    print(f"能量信息统计：")
    print(f"  - 有效能量的构象数量: {len(energies)}")
    print(f"  - 能量为None的构象数量: {energy_none_count}")
    
    if energies:
        print(f"  - 最低能量: {min(energies):.4f} kcal/mol")
        print(f"  - 最高能量: {max(energies):.4f} kcal/mol")
        print(f"  - 能量范围: {max(energies) - min(energies):.4f} kcal/mol")

        if max(energies) - min(energies) < 1e-6:
            print("  - 所有构象能量基本相同!")
        else:
            print("  - 构象间存在能量差异")
    
    if not energies:
        print("策略：没有能量信息，选择收敛的构象")
        logger.debug("No energy information available, selecting converged conformer")
        converged_conformers = [conf for conf in conformers_data if conf.get('converged', False)]
        if converged_conformers:
            print(f"找到 {len(converged_conformers)} 个收敛构象，选择第一个")
            selected_conformer = converged_conformers[0]
            print(f"✓ 选择构象 0 (收敛构象)")
            return selected_conformer
        else:
            print("没有收敛标记，选择第一个构象")
            selected_conformer = conformers_data[0]
            print(f"✓ 选择构象 0 (默认选择)")
            return selected_conformer
    
    print("策略：使用50%能量 + 50%紧凑性评分")

    score_conformers_for_energetic_materials(conformers_data)

    best_conformer = max(conformers_data, key=lambda x: x['total_score'])
    min_energy = min(energies)

    best_index = conformers_data.index(best_conformer)
    
    print(f"✓ 选择构象 {best_index} (综合评分={best_conformer['total_score']:.3f})")
    print(f"最优构象评分详情:")
    print(f"  - 构象索引: {best_index}")
    print(f"  - 能量评分: {best_conformer['energy_score']:.3f}")
    print(f"  - 紧凑性评分: {best_conformer['compactness_score']:.3f}")
    print(f"  - 综合评分: {best_conformer['total_score']:.3f}")
    print(f"  - 实际能量: {best_conformer['energy']:.4f} kcal/mol")
    print(f"  - 相对能量: {best_conformer['energy'] - min_energy:.4f} kcal/mol")

    molecular_volume = calculate_molecular_volume(best_conformer['mol'])
    print(f"  - 分子体积: {molecular_volume:.2f} Ų")
    
    logger.debug(f"Selected conformer scores - Energy: {best_conformer['energy_score']:.3f}, "
                f"Compactness: {best_conformer['compactness_score']:.3f}, "
                f"Total: {best_conformer['total_score']:.3f}, "
                f"Actual Energy: {best_conformer['energy']:.2f} kcal/mol")
    
    return best_conformer

def calculate_compactness_score(mol, coordinates):
    """
    计算构象的紧凑性评分
    
    :param coordinates: 原子坐标
    :return: 紧凑性评分 (0-1)
    """
    try:
        center = np.mean(coordinates, axis=0)
        distances = np.linalg.norm(coordinates - center, axis=1)
        radius_of_gyration = np.sqrt(np.mean(distances ** 2))

        max_distance = np.max(pdist(coordinates))

        if max_distance > 0:
            compactness = radius_of_gyration / max_distance
            score = 1.0 / (1.0 + compactness)
        else:
            score = 1.0

        return min(1.0, max(0.0, score))
    except:
        print("YESSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS\n")
        return 0.5

def calculate_molecular_volume(mol):
    """
    计算分子体积
    
    :param mol: 分子对象
    :return: 分子体积
    """
    try:
        return Descriptors.MolVolume(mol)
    except:
        return 0.0


def fallback_single_conformer(smi, seed, topk_k=3):

    try:
        from .conformer import inner_smi2coords
        mol = inner_smi2coords(smi, seed=seed, remove_hs=False, return_mol=True)
        coords = mol.GetConformer().GetPositions().astype(np.float32)
        coords_energy_list = [(coords, 0.0)] * topk_k
        return mol, coords_energy_list
    except Exception as e:
        logger.error(f"Fallback method also failed for {smi}: {str(e)}, skipping this molecule")
        return None, None
def energy_prefilter_3kcal(conformers_data):
    values = [c['energy'] for c in conformers_data if c.get('energy') is not None]
    if not values:
        return conformers_data
    base = min(values)
    kept = [c for c in conformers_data if c.get('energy') is not None and (c['energy'] - base) <= 3.0]
    if kept:
        logger.info(f"能量预筛: {len(kept)}/{len(conformers_data)} 保留 (ΔE≤3 kcal)")
        return kept
    return conformers_data
