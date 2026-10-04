import unittest

from shopping_grpo.environment.manifest import (
    MANIFEST_VERSION,
    validate_manifest,
)


class EnvironmentManifestTest(unittest.TestCase):
    def test_current_environment_contract_is_validated(self):
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "environment_version": "shopsimulator-environment-v2.1",
            "shopsimulator_commit": "a" * 40,
            "product_data_sha256": "c" * 64,
            "search": {
                "version": "shopsimulator-multifield-bm25-v2",
                "page_size": 20,
            },
            "reward": {"version": "shopsimulator-reward-v3"},
            "observation_version": "shopping-observation-v2",
            "tool_version": "shopping-tools-v2",
            "max_steps": 35,
            "seed": 20260726,
        }
        self.assertIs(validate_manifest(manifest), manifest)

    def test_missing_required_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_manifest({})

    def test_current_environment_requires_reward_v3(self):
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "environment_version": "shopsimulator-environment-v2.1",
            "shopsimulator_commit": "a" * 40,
            "product_data_sha256": "c" * 64,
            "search": {
                "version": "shopsimulator-multifield-bm25-v2",
                "page_size": 20,
            },
            "reward": {"version": "shopsimulator-reward-v3"},
            "observation_version": "shopping-observation-v2",
            "tool_version": "shopping-tools-v2",
            "max_steps": 35,
            "seed": 20260726,
        }
        manifest["reward"] = {"version": "unsupported-reward"}
        with self.assertRaisesRegex(ValueError, "requires shopsimulator-reward-v3"):
            validate_manifest(manifest)

    def test_wrong_tool_contract_is_rejected(self):
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "shopsimulator_commit": "a" * 40,
            "product_data_sha256": "c" * 64,
            "search": {
                "version": "shopsimulator-multifield-bm25-v2",
                "page_size": 20,
            },
            "reward": {"version": "shopsimulator-reward-v3"},
            "observation_version": "shopping-observation-v2",
            "tool_version": "unsupported-tools",
            "max_steps": 35,
            "seed": 20260726,
        }
        with self.assertRaisesRegex(ValueError, "Tool v2"):
            validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
