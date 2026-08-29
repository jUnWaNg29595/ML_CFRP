import pandas as pd
import pytest

from core import pubchem_client
from core.melting_point_data import parse_melting_point_text

def test_split_structure_queries_accepts_newlines_and_semicolons():
    assert pubchem_client.split_structure_queries(' C1OC1\n [O;r3]1[#6;r3][#6;r3]1; C1OC1 ') == [
        'C1OC1',
        '[O;r3]1[#6;r3][#6;r3]1',
    ]


def test_fetch_records_by_smarts_queries_merges_and_deduplicates(monkeypatch, tmp_path):
    monkeypatch.setattr(pubchem_client, 'PUBCHEM_CACHE_DIR', tmp_path)
    calls = []

    def fetch_records(query, **kwargs):
        calls.append(query)
        if query == 'C1OC1':
            return pd.DataFrame([{
                'cid': 1, 'smiles': 'C1OC1', 'mp_raw': '40 °C', 'mp_c': 40.0,
                'component_role': 'resin', 'hardener_class': '',
            }])
        return pd.DataFrame([{
            'cid': 1, 'smiles': 'C1OC1', 'mp_raw': '40 °C', 'mp_c': 40.0,
            'component_role': 'resin', 'hardener_class': '',
        }, {
            'cid': 2, 'smiles': 'CC1OC1', 'mp_raw': '20 °C', 'mp_c': 20.0,
            'component_role': 'resin', 'hardener_class': '',
        }])

    monkeypatch.setattr(pubchem_client, 'fetch_melting_point_records_by_smarts', fetch_records)
    result = pubchem_client.fetch_melting_point_records_by_smarts_queries(
        'C1OC1; [O;r3]1[#6;r3][#6;r3]1', component_role='resin', per_query_max_cids=10,
    )

    assert calls == ['C1OC1', '[O;r3]1[#6;r3][#6;r3]1']
    assert result['cid'].tolist() == [1, 2]
    assert result.attrs['query_count'] == 2


def test_build_cached_melting_point_records_filters_nonindustrial_candidates(monkeypatch, tmp_path):
    monkeypatch.setattr(pubchem_client, 'PUBCHEM_CACHE_DIR', tmp_path)
    pd.DataFrame([
        {'cid': 10, 'smiles': 'C1OC1', 'mol_wt': 58.08},
        {'cid': 11, 'smiles': 'CC1OC1', 'mol_wt': 72.11},
        {'cid': 12, 'smiles': 'c1ccccc1C1OC1', 'mol_wt': 132.16},
        {'cid': 13, 'smiles': 'c1ccccc1CC1OC1', 'mol_wt': 146.19},
        {'cid': 14, 'smiles': 'c1ccccc1CC(C)(C)c2ccc(CC3OC3)cc2', 'mol_wt': 280.0},
        {'cid': 15, 'smiles': 'c1ccccc1.C1OC1', 'mol_wt': 136.0},
    ]).to_csv(tmp_path / 'properties.csv', index=False)
    import json
    annotations = {
        10: '-100 °C', 11: '40 °C', 12: '60 °C', 13: '70 °C', 14: '120 °C', 15: '90 °C',
    }
    for cid, value in annotations.items():
        (tmp_path / f'melting_point_annotation_cid_{cid}_v1.json').write_text(
            json.dumps([{
                'cid': cid, 'mp_raw': value, 'source_url': '', 'source_name': '', 'source_record': cid,
            }]), encoding='utf-8',
        )

    result = pubchem_client.build_cached_melting_point_records(
        'C1OC1', component_role='resin', min_molecular_weight=150.0, max_melting_point_c=500.0,
    )

    assert result['cid'].tolist() == [14]
    assert result.iloc[0]['component_role'] == 'resin'
    assert result.iloc[0]['hardener_class'] == ''


def test_extracts_melting_point_annotation_and_source(monkeypatch, tmp_path):
    monkeypatch.setattr(pubchem_client, "PUBCHEM_CACHE_DIR", tmp_path)
    payload = {
        "Record": {
            "RecordNumber": 123,
            "Section": [{
                "TOCHeading": "Melting Point",
                "Information": [{
                    "Value": {"StringWithMarkup": [{"String": "126 °C"}]},
                    "Reference": [{"URL": "https://example.test/ref"}],
                }],
            }],
        }
    }
    monkeypatch.setattr(pubchem_client, "_request_json", lambda *args, **kwargs: payload)
    result = pubchem_client.fetch_melting_point_annotations_by_cids([123], max_workers=1)
    assert result.iloc[0]["cid"] == 123
    assert result.iloc[0]["mp_raw"] == "126 °C"
    assert result.iloc[0]["source_url"] == "https://example.test/ref"


def test_uses_pug_view_data_endpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(pubchem_client, "PUBCHEM_CACHE_DIR", tmp_path)
    requested_urls = []

    def request_json(url, **kwargs):
        requested_urls.append(url)
        return {"Record": {"Section": []}}

    monkeypatch.setattr(pubchem_client, "_request_json", request_json)
    pubchem_client.fetch_melting_point_annotations_by_cids([123], max_workers=1)
    assert requested_urls == [
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/123/JSON"
    ]


def test_resolves_reference_numbers_and_reference_dictionaries(monkeypatch, tmp_path):
    monkeypatch.setattr(pubchem_client, "PUBCHEM_CACHE_DIR", tmp_path)
    payload = {
        "Record": {
            "RecordNumber": 123,
            "Reference": [{
                "ReferenceNumber": 7,
                "SourceName": "Registry source",
                "SourceURL": "https://example.test/registry",
            }],
            "Section": [{
                "TOCHeading": "Melting Point",
                "Information": [
                    {
                        "Value": {"String": "126 deg C"},
                        "Reference": [7],
                    },
                    {
                        "Value": {"String": "127 deg C"},
                        "Reference": {"URL": "https://example.test/direct", "Name": "Direct source"},
                    },
                ],
            }],
        }
    }
    monkeypatch.setattr(pubchem_client, "_request_json", lambda *args, **kwargs: payload)

    result = pubchem_client.fetch_melting_point_annotations_by_cids([123], max_workers=1)

    assert result[["mp_raw", "source_url", "source_name"]].to_dict("records") == [
        {
            "mp_raw": "126 deg C",
            "source_url": "https://example.test/registry",
            "source_name": "Registry source",
        },
        {
            "mp_raw": "127 deg C",
            "source_url": "https://example.test/direct",
            "source_name": "Direct source",
        },
    ]


def test_extracts_nested_numeric_annotation_and_deduplicates(monkeypatch, tmp_path):
    monkeypatch.setattr(pubchem_client, "PUBCHEM_CACHE_DIR", tmp_path)
    payload = {
        "Record": {
            "RecordNumber": 7,
            "Section": [{
                "TOCHeading": "Physical Description",
                "Section": [{
                    "TOCHeading": "Melting Point",
                    "Information": [
                        {"Value": {"Number": [126], "Unit": "deg C"}},
                        {"Value": {"Number": [126], "Unit": "deg C"}},
                    ],
                }],
            }],
        }
    }
    monkeypatch.setattr(pubchem_client, "_request_json", lambda *args, **kwargs: payload)
    result = pubchem_client.fetch_melting_point_annotations_by_cids([7], max_workers=1)
    assert result["mp_raw"].tolist() == ["126 deg C"]
    assert result.iloc[0]["source_record"] == 7


def test_missing_melting_point_section_returns_empty_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(pubchem_client, "PUBCHEM_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        pubchem_client,
        "_request_json",
        lambda *args, **kwargs: {"Record": {"Section": []}},
    )
    result = pubchem_client.fetch_melting_point_annotations_by_cids([123], max_workers=1)
    assert result.empty
    assert list(result.columns) == [
        "cid", "mp_raw", "source_url", "source_name", "source_record"
    ]


def test_melting_point_annotation_progress_reports_cache_and_request_counts(monkeypatch, tmp_path):
    monkeypatch.setattr(pubchem_client, "PUBCHEM_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        pubchem_client,
        "_request_json",
        lambda *args, **kwargs: {"Record": {"Section": []}},
    )
    events = []

    result = pubchem_client.fetch_melting_point_annotations_by_cids(
        [101, 102],
        max_workers=1,
        progress_callback=events.append,
    )

    assert result.empty
    assert events[0]["total_cids"] == 2
    assert events[-1]["completed_cids"] == 2
    assert events[-1]["cache_hits"] == 0
    assert events[-1]["fetched_cids"] == 2
    assert events[-1]["failed_cids"] == 0
    assert events[-1]["phase"] == "annotations"


def test_property_fetch_progress_reports_completed_cids(monkeypatch):
    monkeypatch.setattr(
        pubchem_client,
        "_request_json",
        lambda *args, **kwargs: {
            "PropertyTable": {
                "Properties": [
                    {"CID": 1, "CanonicalSMILES": "CCO"},
                    {"CID": 2, "CanonicalSMILES": "CCN"},
                ]
            }
        },
    )
    events = []

    result = pubchem_client.fetch_properties_by_cids(
        [1, 2],
        properties=["CanonicalSMILES"],
        max_workers=1,
        progress_callback=events.append,
    )

    assert len(result) == 2
    assert events[-1]["phase"] == "properties"
    assert events[-1]["completed_cids"] == 2
    assert events[-1]["total_cids"] == 2


def test_query_records_keep_role_and_class(monkeypatch, tmp_path):
    monkeypatch.setattr(pubchem_client, "PUBCHEM_CACHE_DIR", tmp_path)
    monkeypatch.setattr(pubchem_client, "fetch_cids_by_smarts", lambda *args, **kwargs: [11])
    monkeypatch.setattr(
        pubchem_client,
        "fetch_properties_by_cids",
        lambda *args, **kwargs: pd.DataFrame([{
            "CID": 11,
            "CanonicalSMILES": "CCO",
            "MolecularWeight": "46.07",
        }]),
    )
    monkeypatch.setattr(
        pubchem_client,
        "fetch_melting_point_annotations_by_cids",
        lambda *args, **kwargs: pd.DataFrame([{
            "cid": 11, "mp_raw": "-114 °C", "source_url": "https://example.test/mp"
        }]),
    )
    result = pubchem_client.fetch_melting_point_records_by_smarts(
        "[OX2H]", component_role="hardener", hardener_class="酚", max_cids=10,
    )
    assert result.iloc[0]["component_role"] == "hardener"
    assert result.iloc[0]["hardener_class"] == "酚"
    assert result.iloc[0]["mp_c"] == -114
    assert result.iloc[0]["smiles"] == "CCO"
    assert result.iloc[0]["mol_wt"] == 46.07
    assert result.iloc[0]["canonical_smiles"] == "CCO"



def test_melting_point_query_revalidates_pubchem_structures(monkeypatch, tmp_path):
    monkeypatch.setattr(pubchem_client, "PUBCHEM_CACHE_DIR", tmp_path)
    monkeypatch.setattr(pubchem_client, "fetch_cids_by_smarts", lambda *args, **kwargs: [11, 12])
    monkeypatch.setattr(
        pubchem_client,
        "fetch_properties_by_cids",
        lambda *args, **kwargs: pd.DataFrame([
            {"CID": 11, "CanonicalSMILES": "CCO", "MolecularWeight": "46.07"},
            {"CID": 12, "CanonicalSMILES": "CCC", "MolecularWeight": "44.10"},
        ]),
    )
    monkeypatch.setattr(
        pubchem_client,
        "fetch_melting_point_annotations_by_cids",
        lambda *args, **kwargs: pd.DataFrame([
            {"cid": 11, "mp_raw": "-114 °C"},
            {"cid": 12, "mp_raw": "-188 °C"},
        ]),
    )

    result = pubchem_client.fetch_melting_point_records_by_smarts(
        "[OX2H]", component_role="hardener", max_cids=10,
    )

    assert result["cid"].tolist() == [11]
    assert result.attrs["query_validated"] is True
    assert result.attrs["query_rejected_count"] == 1


def test_cached_melting_point_query_revalidates_structures(monkeypatch, tmp_path):
    monkeypatch.setattr(pubchem_client, "PUBCHEM_CACHE_DIR", tmp_path)
    cache_payload = {
        "query": "[OX2H]",
        "component_role": "resin",
        "hardener_class": "",
        "max_cids": 10,
    }
    cache_path = pubchem_client._melting_point_cache_path("records", cache_payload)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"cid": 11, "smiles": "CCO", "mp_raw": "-114 °C"},
        {"cid": 12, "smiles": "CCC", "mp_raw": "-188 °C"},
    ]).to_csv(cache_path, index=False)

    result = pubchem_client.fetch_melting_point_records_by_smarts(
        "[OX2H]", component_role="resin", max_cids=10,
    )

    assert result["cid"].tolist() == [11]
    assert result.attrs["cache_source"] == "disk"

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("126 deg C", 126.0),
        ("212 deg F", 100.0),
        ("273.15 deg K", 0.0),
        ("32 degrees Fahrenheit", 0.0),
    ],
)
def test_parses_degree_units_to_celsius(raw, expected):
    result = parse_melting_point_text(raw)
    assert result["mp_c"] == pytest.approx(expected)
    assert result["mp_quality"] == "high"


def test_parses_ranges_with_degree_units_to_celsius():
    result = parse_melting_point_text("32 deg F to 212 deg F")
    assert result["mp_lower_c"] == pytest.approx(0.0)
    assert result["mp_upper_c"] == pytest.approx(100.0)
    assert result["mp_quality"] == "range"


def test_smiles_fallback_uses_non_empty_values_in_order(monkeypatch):
    monkeypatch.setattr(pubchem_client, "fetch_cids_by_smarts", lambda *args, **kwargs: [1, 2, 3, 4])
    requested_properties = []

    def fetch_properties(cids, properties, **kwargs):
        requested_properties.append(list(properties))
        return pd.DataFrame([
            {"CID": 1, "CanonicalSMILES": "", "ConnectivitySMILES": "CCO"},
            {"CID": 2, "CanonicalSMILES": None, "ConnectivitySMILES": "", "IsomericSMILES": "CCN"},
            {"CID": 3, "CanonicalSMILES": None, "ConnectivitySMILES": None, "IsomericSMILES": "", "SMILES": "CCC"},
            {"CID": 4, "CanonicalSMILES": None, "ConnectivitySMILES": None, "IsomericSMILES": None, "SMILES": ""},
        ])

    monkeypatch.setattr(pubchem_client, "fetch_properties_by_cids", fetch_properties)
    result = pubchem_client._fetch_smiles_by_smarts_uncached("[OX2H]", property_workers=1)

    assert requested_properties == [[
        "CanonicalSMILES", "ConnectivitySMILES", "IsomericSMILES", "SMILES", "MolecularWeight"
    ]]
    assert result["smiles"].tolist() == ["CCO", "CCN", "CCC"]


def test_end_to_end_records_use_smiles_fallback_and_degree_unit(monkeypatch, tmp_path):
    monkeypatch.setattr(pubchem_client, "PUBCHEM_CACHE_DIR", tmp_path)
    monkeypatch.setattr(pubchem_client, "fetch_cids_by_smarts", lambda *args, **kwargs: [11])
    monkeypatch.setattr(
        pubchem_client,
        "fetch_properties_by_cids",
        lambda *args, **kwargs: pd.DataFrame([{
            "CID": 11,
            "CanonicalSMILES": "",
            "ConnectivitySMILES": None,
            "IsomericSMILES": "CCO",
            "SMILES": "CCC",
            "MolecularWeight": "46.07",
        }]),
    )
    monkeypatch.setattr(
        pubchem_client,
        "fetch_melting_point_annotations_by_cids",
        lambda *args, **kwargs: pd.DataFrame([{
            "cid": 11, "mp_raw": "32 deg F", "source_url": "https://example.test/mp"
        }]),
    )

    result = pubchem_client.fetch_melting_point_records_by_smarts(
        "[OX2H]", component_role="resin", max_cids=10,
    )

    assert result.iloc[0]["smiles"] == "CCO"
    assert result.iloc[0]["mp_c"] == pytest.approx(0.0)


def test_non_positive_max_cids_short_circuits_safely(monkeypatch, tmp_path):
    monkeypatch.setattr(pubchem_client, "PUBCHEM_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        pubchem_client,
        "_request_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network should not be called")),
    )

    assert pubchem_client.fetch_cids_by_smarts("[OX2H]", max_cids=0) == []
    assert pubchem_client.fetch_cids_by_smiles("CCO", max_cids=-1) == []
    assert pubchem_client.fetch_smiles_by_smarts("[OX2H]", max_cids=0).empty
    assert pubchem_client.fetch_melting_point_records_by_smarts(
        "[OX2H]", component_role="resin", max_cids=-1
    ).empty


def test_empty_annotation_input_has_stable_schema(monkeypatch, tmp_path):
    monkeypatch.setattr(pubchem_client, "PUBCHEM_CACHE_DIR", tmp_path)
    result = pubchem_client.fetch_melting_point_annotations_by_cids([], max_workers=1)
    assert result.empty
    assert list(result.columns) == pubchem_client._MELTING_POINT_ANNOTATION_COLUMNS


def test_partial_cid_failure_retries_only_missing_cid(monkeypatch, tmp_path):
    monkeypatch.setattr(pubchem_client, "PUBCHEM_CACHE_DIR", tmp_path)
    calls = []
    failed_once = {2: True}

    def request_json(url, **kwargs):
        cid = int(url.split('/data/compound/')[1].split('/')[0])
        calls.append(cid)
        if failed_once.get(cid):
            failed_once[cid] = False
            raise RuntimeError("temporary failure")
        return {
            "Record": {
                "RecordNumber": cid,
                "Section": [{
                    "TOCHeading": "Melting Point",
                    "Information": [{"Value": {"String": f"{100 + cid} deg C"}}],
                }],
            }
        }

    monkeypatch.setattr(pubchem_client, "_request_json", request_json)
    first = pubchem_client.fetch_melting_point_annotations_by_cids([1, 2], max_workers=1)
    assert first.attrs["failed_cids"] == [2]
    assert calls == [1, 2]

    second = pubchem_client.fetch_melting_point_annotations_by_cids([1, 2], max_workers=1)
    assert second.attrs["failed_cids"] == []
    assert calls == [1, 2, 2]
    assert set(second["cid"].tolist()) == {1, 2}
