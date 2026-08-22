import pandas as pd

from core.molecule_design import (
    DesignConfig,
    DesignProduct,
    DesignResult,
    Scaffold,
    ScaffoldMiner,
    SearchConfig,
    compute_design_hash,
)


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
