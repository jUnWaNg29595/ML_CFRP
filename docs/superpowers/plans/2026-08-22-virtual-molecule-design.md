# Virtual Molecule Design Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the virtual screening page's random/prebuilt candidate flow with a model-driven molecular design engine that creates chemically connected, auditable variants from validated scaffolds.

**Architecture:** Add a focused `core/molecule_design.py` domain module for scaffold mining, versioned Reaction SMARTS templates, direct RDKit edits, constrained beam search, validation, scoring, and design traces. Keep existing feature-workflow/model prediction helpers as the scoring boundary, then replace the old `page_virtual_screening` candidate-source flow with a design-only Streamlit workflow.

**Tech Stack:** Python 3.13, RDKit, pandas, NumPy, existing Streamlit app, pytest, existing `core.virtual_screening` prediction/feature-contract utilities.

## Global Constraints

- Use validated training/candidate molecules as scaffolds; no parentless free generation.
- Implement A (scaffold edits), B (Reaction SMARTS templates), and C (model-guided constrained beam search).
- Default-enable aryl substitution, OH/NH2 functionalization, C1-C6 chain scans, and role-specific resin/hardener rules; every category remains independently configurable.
- Validate every product with RDKit sanitize, allowed elements, bond order, valence, ring closure, duplicate-connection, and functional-group checks.
- Resin and hardener are designed separately and must never be covalently joined as one product molecule.
- Missing molecular features or incompatible model contracts stop prediction; do not fill with zero, NaN, baseline, or guessed values.
- Remove the old direct candidate-library/random/prebuilt molecule entry points from the virtual screening page.
- Write tests first, observe the expected failure, then implement the minimum code to pass.
- Preserve existing model import, molecular-feature workflow, applicability-domain, and export contracts.

---

### Task 1: Add the molecular-design domain types and scaffold miner

**Files:**
- Create: `core/molecule_design.py`
- Create: `tests/test_molecule_design.py`
- Modify: `core/__init__.py` only if the project exports core modules there

**Interfaces:**
- Produces `DesignConfig`, `SearchConfig`, `Scaffold`, `DesignProduct`, `DesignResult`, `ScaffoldMiner`, and `compute_design_hash` for later tasks.
- `ScaffoldMiner.from_frame(frame, role, smiles_columns, max_scaffolds, random_state) -> list[Scaffold]` must normalize SMILES with the existing `core.smiles_utils` parser, preserve source row/index, and deduplicate canonical structures in stable order.

- [ ] **Step 1: Write failing tests for typed configuration, stable hashing, and scaffold extraction**

```python
def test_scaffold_miner_keeps_valid_training_scaffolds_in_order():
    frame = pd.DataFrame({"smiles": ["C1CO1", "invalid", "C1CO1", "NCCN"]})
    scaffolds = ScaffoldMiner.from_frame(
        frame, role="resin", smiles_columns=["smiles"], max_scaffolds=10, random_state=7
    )
    assert [item.smiles for item in scaffolds] == ["C1CO1", "NCCN"]
    assert all(item.role == "resin" for item in scaffolds)

def test_design_hash_changes_when_any_config_parameter_changes():
    first = compute_design_hash(DesignConfig(random_state=42))
    second = compute_design_hash(DesignConfig(random_state=43))
    assert first != second
```

- [ ] **Step 2: Run the focused tests and verify the expected missing-symbol failures**

Run: `pytest tests/test_molecule_design.py -q`

Expected: FAIL because the new module and public types do not exist yet.

- [ ] **Step 3: Implement the minimal domain model**

Use dataclasses with JSON-safe fields. `DesignProduct` must contain `parent_smiles`, `product_smiles`, `role`, `design_method`, `template_id`, `edit_trace`, `design_depth`, `chemical_validity`, `filter_reason`, and optional scoring fields. `compute_design_hash` must serialize dataclasses with sorted keys and hash the UTF-8 JSON using SHA-256. `ScaffoldMiner` must call the existing parser, skip invalid/empty values, and preserve first-seen order.

- [ ] **Step 4: Run focused tests and verify they pass**

Run: `pytest tests/test_molecule_design.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the domain model**

```bash
git add core/molecule_design.py tests/test_molecule_design.py core/__init__.py
git commit -m "feat: add molecular design domain model"
```

### Task 2: Implement Reaction SMARTS registry and chemically connected A/B edits

**Files:**
- Modify: `core/molecule_design.py`
- Modify: `tests/test_molecule_design.py`

**Interfaces:**
- Produces `ReactionTemplate`, `ReactionTemplateRegistry`, `apply_design_template`, `validate_product`, and `generate_rule_based_variants`.
- `apply_design_template(smiles, template_id, role) -> list[DesignProduct]` must return complete connected product SMILES plus atom-map/connection audit; it must never return a dot-disconnected resin/hardener pair.

- [ ] **Step 1: Write failing tests for all four edit families and invalid valence rejection**

```python
@pytest.mark.parametrize("template_id, scaffold", [
    ("aryl_methyl_substitution", "c1ccccc1"),
    ("hydroxyl_glycidyl_ether", "Oc1ccccc1"),
    ("amine_alkylation", "NCCN"),
    ("ether_chain_scan", "CCOCC1CO1"),
])
def test_templates_create_single_connected_valid_products(template_id, scaffold):
    role = "hardener" if template_id == "amine_alkylation" else "resin"
    products = apply_design_template(scaffold, template_id, role=role)
    assert products
    for product in products:
        assert "." not in product.product_smiles
        assert product.chemical_validity is True
        assert validate_product(product.product_smiles, role=role).ok

def test_template_rejects_a_second_substitution_on_saturated_site():
    products = apply_design_template("C", "aryl_methyl_substitution", role="resin")
    assert products == []
```

- [ ] **Step 2: Run the tests and verify they fail for missing templates/validation**

Run: `pytest tests/test_molecule_design.py -k "template or connected or saturated" -q`

Expected: FAIL because the registry and edit executor are not implemented.

- [ ] **Step 3: Implement versioned templates and direct RDKit connection logic**

Define explicit low-risk substituent fragments with a dummy attachment atom, Reaction SMARTS metadata, role allowlists, and product limits. Use RDKit `ChemicalReaction` where a template is expressible; use an `RWMol` helper for mapped attachment when the template needs explicit aromatic-H/OH/NH2 site selection. Every generated product must be sanitized, canonicalized, checked for a single connected component, and passed through `validate_product`. Record parent atom index, new atom index, bond type, template version, and rejection reason.

- [ ] **Step 4: Run focused tests and verify all four edit families pass**

Run: `pytest tests/test_molecule_design.py -k "template or connected or saturated" -q`

Expected: PASS.

- [ ] **Step 5: Commit the template engine**

```bash
git add core/molecule_design.py tests/test_molecule_design.py
git commit -m "feat: add validated molecular edit templates"
```

### Task 3: Add role-specific validation and rule-based variant generation

**Files:**
- Modify: `core/molecule_design.py`
- Modify: `tests/test_molecule_design.py`

**Interfaces:**
- Produces `validate_product` with a structured `ValidationReport` and `generate_rule_based_variants(scaffolds, config) -> list[DesignProduct]`.
- Uses the existing epoxy/active-site rule semantics from `core.virtual_screening` without duplicating prediction logic.

- [ ] **Step 1: Write failing tests for role separation and functional-group limits**

```python
def test_resin_template_cannot_apply_hardener_only_edit():
    assert apply_design_template("NCCN", "hydroxyl_glycidyl_ether", role="hardener") == []

def test_generated_resin_keeps_epoxide_role_and_hardener_keeps_active_site_role():
    resin = generate_rule_based_variants(
        [Scaffold("C1CO1", "resin", "train", 0)],
        DesignConfig(enabled_templates=["ether_chain_scan"]),
    )
    hardener = generate_rule_based_variants(
        [Scaffold("NCCN", "hardener", "train", 1)],
        DesignConfig(enabled_templates=["amine_alkylation"]),
    )
    assert all(validate_product(x.product_smiles, "resin").role_valid for x in resin)
    assert all(validate_product(x.product_smiles, "hardener").role_valid for x in hardener)
```

- [ ] **Step 2: Run the role tests and verify the expected failures**

Run: `pytest tests/test_molecule_design.py -k "role or functional" -q`

Expected: FAIL because role-aware generation and functional-group checks are incomplete.

- [ ] **Step 3: Implement role rules, limits, and deterministic per-scaffold quotas**

Use the existing `DEFAULT_EPOXY_RULES`, `_calc_rule_features`, and `filter_candidates_by_epoxy_rules` semantics through small adapter functions. Apply resin epoxide and hardener active-site constraints after every edit. Enforce per-template and per-scaffold limits before returning products; retain the unedited parent when `keep_parents=True`.

- [ ] **Step 4: Run the role tests and the full molecule-design test file**

Run: `pytest tests/test_molecule_design.py -q`

Expected: PASS.

- [ ] **Step 5: Commit role-aware generation**

```bash
git add core/molecule_design.py tests/test_molecule_design.py
git commit -m "feat: enforce role-aware molecular design rules"
```

### Task 4: Implement model-guided constrained beam search and scoring adapter

**Files:**
- Modify: `core/molecule_design.py`
- Modify: `tests/test_molecule_design.py`

**Interfaces:**
- Produces `ModelGuidedGraphSearch`, `search_design_space`, and `score_design_products`.
- Produces `design_molecules(scaffolds, config, *, model=None, pipeline=None, feature_cols=None, scorer=None) -> DesignResult` as the orchestration entry point used by the page.
- `score_design_products` must accept the existing model/pipeline/feature columns and call `core.virtual_screening.extract_features_from_config`, `build_feature_matrix`, `predict_with_model`, and contract validation where applicable.

- [ ] **Step 1: Write failing tests for deterministic search and model-based ranking**

```python
class FakeModel:
    def predict(self, values):
        return values["design_value"].to_numpy()

def test_beam_search_is_seed_stable_and_uses_model_score():
    seeds = [DesignProduct("C1CO1", "C1CO1", "resin", "parent", "", [], 0, True)]
    config = SearchConfig(depth=2, beam_width=3, random_state=11)
    first = search_design_space(seeds, config, scorer=lambda items: [1.0 + i for i, _ in enumerate(items)])
    second = search_design_space(seeds, config, scorer=lambda items: [1.0 + i for i, _ in enumerate(items)])
    assert [x.product_smiles for x in first] == [x.product_smiles for x in second]
    assert first[0].model_score >= first[-1].model_score
```

- [ ] **Step 2: Run the search tests and verify they fail**

Run: `pytest tests/test_molecule_design.py -k "beam or model_score" -q`

Expected: FAIL because the search and scoring adapter do not exist.

- [ ] **Step 3: Implement beam search and scoring**

Expand only valid products from the template registry. Use a stable sort key `(score, canonical_smiles, template_id)` and a seeded tie-breaker. Keep top `beam_width` exploitation candidates plus `exploration_ratio` candidates that pass fingerprint distance and parent/template diversity constraints. Record prediction, prediction standard deviation/source, applicability score, synth score, and score components. Stop before prediction when feature extraction or contract validation fails.

- [ ] **Step 4: Run focused and regression tests**

Run: `pytest tests/test_molecule_design.py tests/test_virtual_screening.py -q`

Expected: PASS.

- [ ] **Step 5: Commit model-guided search**

```bash
git add core/molecule_design.py tests/test_molecule_design.py
git commit -m "feat: add model-guided molecular beam search"
```

### Task 5: Replace the old virtual-screening candidate flow with the design engine

**Files:**
- Modify: `app.py` in `page_virtual_screening` and its local imports/state keys
- Modify: `core/virtual_screening.py` only for thin compatibility adapters if needed
- Modify: `tests/test_virtual_screening.py`

**Interfaces:**
- The page consumes `ScaffoldMiner`, `DesignConfig`, `SearchConfig`, and `design_molecules`.
- The page stores `vs_design_config`, `vs_design_preview`, `vs_design_result_df`, `vs_design_trace`, and `vs_design_hash` in session state.
- No old `vs_formula_*` direct candidate-source controls remain reachable from the page.

- [ ] **Step 1: Write failing page/state tests**

```python
def test_virtual_screening_page_exposes_design_engine_only(monkeypatch):
    import app
    source = Path(app.__file__).read_text(encoding="utf-8")
    page_source = source[source.index("def page_virtual_screening"):]
    assert "分子设计引擎" in page_source
    assert "叠加 PubChem 候选" not in page_source
    assert "虚拟完整分子上限" not in page_source
```

- [ ] **Step 2: Run the page test and verify the old-entrypoint failure**

Run: `pytest tests/test_virtual_screening.py -k "design_engine_only" -q`

Expected: FAIL because the current page still contains the old candidate-source controls.

- [ ] **Step 3: Replace the page flow**

Use the existing model import and workflow resolution at the top of `page_virtual_screening`; replace the candidate-source/formulation controls with: role-specific scaffold column selectors, four edit-category toggles and limits, template multiselect with SMARTS/risk previews, beam-search settings, preview button, run button, trace/result tables, and CSV downloads. On every config change, compute `vs_design_hash` and clear preview/results whose hash differs. Feed the generated resin and hardener products into the existing feature/prediction/evaluation path only through the new design result; do not expose the previous random/generated/PubChem candidate controls.

- [ ] **Step 4: Run page/state and existing navigation tests**

Run: `pytest tests/test_virtual_screening.py tests/test_navigation.py tests/test_prediction_portal.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the page migration**

```bash
git add app.py core/virtual_screening.py tests/test_virtual_screening.py
git commit -m "feat: replace virtual screening candidates with molecular design"
```

### Task 6: Add persistence, preview auditing, and end-to-end regression coverage

**Files:**
- Modify: `core/molecule_design.py`
- Modify: `app.py`
- Modify: `tests/test_molecule_design.py`
- Modify: `tests/test_virtual_screening.py`

**Interfaces:**
- `MoleculeDesignResult.to_frame()` returns a stable, export-ready DataFrame with all trace and score columns.
- `MoleculeDesignResult.to_dict()` / `from_dict()` preserve template versions, config, random seed, design hash, products, failures, and score metadata.

- [ ] **Step 1: Write failing persistence and failure-audit tests**

```python
def test_design_result_round_trip_preserves_trace_and_failures():
    result = design_molecules(
        [Scaffold("C1CO1", "resin", "train", 0)],
        DesignConfig(enabled_templates=["ether_chain_scan"], random_state=5),
    )
    restored = MoleculeDesignResult.from_dict(result.to_dict())
    assert restored.design_hash == result.design_hash
    assert restored.to_frame().equals(result.to_frame())
    assert restored.failures == result.failures

def test_all_failed_scaffolds_block_prediction_without_fabricated_features():
    result = design_molecules(
        [],
        DesignConfig(enabled_templates=["aryl_methyl_substitution"], random_state=5),
    )
    assert result.can_predict is False
    assert result.prediction_block_reason
```

- [ ] **Step 2: Run the tests and verify the missing serialization/blocking behavior**

Run: `pytest tests/test_molecule_design.py -k "round_trip or fabricated or block" -q`

Expected: FAIL until result serialization and prediction blocking are implemented.

- [ ] **Step 3: Implement serialization and Streamlit preview/result state**

Serialize only JSON-safe values, preserve canonical product order, and validate the saved design hash on load. Display stage counts (`scaffolds`, `template_products`, `valid_products`, `scored_products`, `returned_products`) and failure reasons in the preview. Disable the final run button when no valid products or when the feature contract is incomplete.

- [ ] **Step 4: Run the complete relevant suite and a syntax check**

Run:

```bash
pytest tests/test_molecule_design.py tests/test_virtual_screening.py tests/test_navigation.py tests/test_prediction_feature_contract.py -q
python -m py_compile core/molecule_design.py app.py
```

Expected: all tests PASS and `py_compile` exits with code 0.

- [ ] **Step 5: Commit persistence and regression coverage**

```bash
git add core/molecule_design.py app.py tests/test_molecule_design.py tests/test_virtual_screening.py
git commit -m "test: verify auditable virtual molecule design workflow"
```

### Task 7: Run visual/manual acceptance and final verification

**Files:**
- Modify only if verification finds a concrete defect in `app.py`, `core/molecule_design.py`, or focused tests

- [ ] **Step 1: Start the Streamlit app on an unused port**

Run: `streamlit run app.py --server.port 8515`

Expected: the app starts without import errors. Use another unused port if 8515 is occupied.

- [ ] **Step 2: Exercise the design-only flow**

Load an existing model artifact, select training SMILES columns for resin and hardener, run preview with one scaffold, verify product structures are connected, inspect the edit trace, then run the full design. Confirm no old candidate-source controls appear and the result CSV contains parent/product/template/score columns.

- [ ] **Step 3: Run the final verification suite**

Run: `pytest -q`

Expected: the full repository test suite passes, with any pre-existing unrelated failures recorded rather than hidden.

- [ ] **Step 4: Check repository state and summarize evidence**

Run: `git status --short --branch` and `git log -7 --oneline`.

Expected: only intended implementation/test/doc changes are present, and the final response cites the passing test commands and any residual environment limitation.
