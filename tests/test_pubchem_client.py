import unittest

import pandas as pd

from core.pubchem_client import _filter_pubchem_query_matches


class PubChemValidationTests(unittest.TestCase):
    def test_plain_smiles_substructure_results_are_revalidated(self):
        source = pd.DataFrame(
            {
                "cid": [1, 2, 3],
                "smiles": ["CCN(CC)CC", "CCN(CC)CCO", "c1ccccc1"],
                "mol_wt": [101.2, 117.2, 78.1],
            }
        )
        result = _filter_pubchem_query_matches(source, "CCN(CC)CC")
        self.assertEqual(result["cid"].tolist(), [1, 2])
        self.assertTrue(result.attrs.get("query_validated"))
        self.assertEqual(result.attrs.get("rejected_count"), 1)

    def test_smarts_results_are_revalidated(self):
        source = pd.DataFrame(
            {
                "cid": [1, 2],
                "smiles": ["C1CO1", "CCO"],
                "mol_wt": [44.1, 46.1],
            }
        )
        result = _filter_pubchem_query_matches(source, "[O;r3]1[C;r3][C;r3]1")
        self.assertEqual(result["cid"].tolist(), [1])


if __name__ == "__main__":
    unittest.main()
