import pandas as pd
import os
import subprocess
import sys

import pytest

from core.molecule_design import (
    DesignConfig,
    DesignProduct,
    DesignResult,
    Scaffold,
    ScaffoldMiner,
    SearchConfig,
    compute_design_hash,
)


@pytest.mark.parametrize("template_id, scaffold", [
    ("aryl_methyl_substitution", "c1ccccc1"),
    ("hydroxyl_glycidyl_ether", "Oc1ccccc1"),
    ("amine_alkylation", "NCCN"),
    ("ether_chain_scan", "CCOCC1CO1"),
])
def test_templates_create_single_connected_valid_products(template_id, scaffold):
    from core.molecule_design import apply_design_template, validate_product

    role = "hardener" if template_id == "amine_alkylation" else "resin"
    products = apply_design_template(scaffold, template_id, role=role)
    assert products
    for product in products:
        assert "." not in product.product_smiles
        assert product.chemical_validity is True
        assert validate_product(product.product_smiles, role=role).ok


def test_template_rejects_a_second_substitution_on_saturated_site():
    from core.molecule_design import apply_design_template

    assert apply_design_template("C", "aryl_methyl_substitution", role="resin") == []


def test_resin_template_cannot_apply_hardener_only_edit():
    from core.molecule_design import apply_design_template

    assert apply_design_template("NCCN", "hydroxyl_glycidyl_ether", role="hardener") == []


def test_generated_resin_and_hardener_keep_role_specific_functionality():
    from core.molecule_design import (
        DesignConfig,
        Scaffold,
        generate_rule_based_variants,
        validate_product,
    )

    resin = generate_rule_based_variants(
        [Scaffold("C1CO1", "resin", "train", 0)],
        DesignConfig(enabled_templates=["ether_chain_scan"], keep_parents=True),
    )
    hardener = generate_rule_based_variants(
        [Scaffold("NCCN", "hardener", "train", 1)],
        DesignConfig(enabled_templates=["amine_alkylation"], keep_parents=True),
    )
    assert resin and hardener
    assert all(validate_product(x.product_smiles, "resin").role_valid for x in resin)
    assert all(validate_product(x.product_smiles, "hardener").role_valid for x in hardener)


def test_beam_search_is_seed_stable_and_uses_model_score():
    from core.molecule_design import DesignProduct, SearchConfig, search_design_space

    seeds = [DesignProduct("C1CO1", "C1CO1", "resin", "parent", "", [], 0, True)]
    config = SearchConfig(depth=2, beam_width=3, random_state=11)

    def scorer(items):
        return [1.0 + i for i, _ in enumerate(items)]

    first = search_design_space(seeds, config, scorer=scorer)
    second = search_design_space(seeds, config, scorer=scorer)
    assert [x.product_smiles for x in first] == [x.product_smiles for x in second]
    assert first[0].model_score >= first[-1].model_score


def test_scaffold_miner_keeps_valid_training_scaffolds_in_order():
    frame = pd.DataFrame({"smiles": ["C1CO1", "invalid", "C1CO1", "NCCN"]})

    scaffolds = ScaffoldMiner.from_frame(
        frame, role="resin", smiles_columns=["smiles"], max_scaffolds=10, random_state=7
    )

    assert [item.smiles for item in scaffolds] == ["C1CO1", "NCCN"]
    assert [item.source_index for item in scaffolds] == [0, 3]
    assert all(item.role == "resin" for item in scaffolds)


def test_design_hash_changes_when_any_config_parameter_changes():
    first = compute_design_hash(DesignConfig(random_state=42))
    second = compute_design_hash(DesignConfig(random_state=43))

    assert first != second
    assert len(first) == 64


def test_domain_types_have_json_safe_defaults_and_nested_hashing():
    config = DesignConfig(enabled_templates=["ether_chain_scan"])
    product = DesignProduct(
        parent_smiles="C1CO1",
        product_smiles="C1CO1",
        role="resin",
        design_method="parent",
        template_id="",
        edit_trace=[],
        design_depth=0,
        chemical_validity=True,
        filter_reason=None,
    )
    result = DesignResult(products=[product], config=config)

    assert SearchConfig().beam_width > 0
    assert compute_design_hash(result)


def test_design_hash_is_stable_for_nested_sets_across_python_processes():
    script = (
        "from core.molecule_design import compute_design_hash; "
        "print(compute_design_hash({'nested': {'alpha', 'beta', 'gamma', 'delta'}}))"
    )
    hashes = []
    for seed in ("1", "2"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        hashes.append(completed.stdout.strip())

    assert hashes[0] == hashes[1]
