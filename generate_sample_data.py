# -*- coding: utf-8 -*-
"""示例数据生成脚本"""

import pandas as pd
import numpy as np
import os


def generate_hybrid_dataset(n_samples=500, seed=42):
    """生成包含工艺参数和分子SMILES的混合数据集"""
    np.random.seed(seed)

    data = {
        'fiber_volume_ratio': np.random.uniform(0.50, 0.70, n_samples),
        'cure_temperature_C': np.random.uniform(120, 180, n_samples),
        'cure_pressure_MPa': np.random.uniform(0.5, 1.5, n_samples),
        'porosity_percent': np.random.uniform(0.1, 3.0, n_samples),
    }

    resin_smiles = [
        "C1=CC=C(C=C1)C(C2=CC=C(C=C2)O)C",
        "C1=CC=C(C=C1)C(C)(C)C2=CC=C(C=C2)O",
        "C1C(O1)CO.C1=CC(=CC=C1C(C2=CC=C(C=C2)O)C3=CC=C(C=C3)O)O",
        "C(C(F)(F)F)OC1=CC=C(C=C1)C=C",
        "C1=CC=C2C(=C1)C(=O)NC2=O",
    ]

    smiles_strength_factor = {smi: 100 * (i + 1) for i, smi in enumerate(resin_smiles)}
    smiles_list = np.random.choice(resin_smiles, n_samples)
    data['resin_smiles'] = smiles_list

    smiles_effect = np.array([smiles_strength_factor[smi] for smi in smiles_list])

    data['tensile_strength_MPa'] = (
        2500 * data['fiber_volume_ratio'] +
        5 * data['cure_temperature_C'] +
        200 * data['cure_pressure_MPa'] -
        150 * data['porosity_percent'] +
        smiles_effect +
        np.random.normal(0, 80, n_samples)
    )

    df = pd.DataFrame(data)

    for col in ['cure_temperature_C', 'porosity_percent']:
        idx = df.sample(frac=0.05).index
        df.loc[idx, col] = np.nan

    return df


def generate_pure_numeric_dataset(n_samples=500, seed=42):
    """生成纯数值数据集"""
    np.random.seed(seed)

    data = {
        'fiber_volume_ratio': np.random.uniform(0.50, 0.70, n_samples),
        'cure_temperature_C': np.random.uniform(120, 180, n_samples),
        'cure_pressure_MPa': np.random.uniform(0.5, 1.5, n_samples),
        'porosity_percent': np.random.uniform(0.1, 3.0, n_samples),
        'fiber_orientation_deg': np.random.uniform(0, 90, n_samples),
        'resin_viscosity_Pa_s': np.random.uniform(0.1, 10, n_samples),
        'cure_time_min': np.random.uniform(30, 180, n_samples),
    }

    data['tensile_strength_MPa'] = (
        2500 * data['fiber_volume_ratio'] +
        5 * data['cure_temperature_C'] +
        200 * data['cure_pressure_MPa'] -
        150 * data['porosity_percent'] -
        2 * data['fiber_orientation_deg'] +
        10 * data['resin_viscosity_Pa_s'] +
        0.5 * data['cure_time_min'] +
        np.random.normal(0, 50, n_samples)
    )

    data['elastic_modulus_GPa'] = (
        150 * data['fiber_volume_ratio'] +
        0.1 * data['cure_temperature_C'] -
        5 * data['porosity_percent'] +
        np.random.normal(0, 5, n_samples)
    )

    df = pd.DataFrame(data)

    for col in ['cure_temperature_C', 'porosity_percent']:
        idx = df.sample(frac=0.03).index
        df.loc[idx, col] = np.nan

    return df
