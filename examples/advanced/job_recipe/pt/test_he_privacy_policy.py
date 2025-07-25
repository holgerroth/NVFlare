#!/usr/bin/env python3
"""
Test script to verify that HEPrivacyPolicy works correctly.
This script tests the creation of HE privacy filters without actually running FL.
"""

import os
import sys

# Add the nvflare path to sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from nvflare.app_opt.he.model_decryptor import HEModelDecryptor
from nvflare.app_opt.he.model_encryptor import HEModelEncryptor
from nvflare.recipes.fedavg import FedAvgRecipe, HEPrivacyPolicy


def test_he_privacy_policy():
    """Test the HEPrivacyPolicy class."""
    print("Testing HEPrivacyPolicy...")

    # Test basic HE privacy policy
    he_policy = HEPrivacyPolicy(
        tenseal_context_file="test_context.tenseal", encrypt_layers=None, weigh_by_local_iter=True
    )

    print(f"✓ Created HEPrivacyPolicy with tenseal_context_file: {he_policy.tenseal_context_file}")
    print(f"✓ encrypt_layers: {he_policy.encrypt_layers}")
    print(f"✓ weigh_by_local_iter: {he_policy.weigh_by_local_iter}")

    # Test creating encryption filter
    try:
        encrypt_filter = he_policy.create_encrypt_filter()
        assert isinstance(encrypt_filter, HEModelEncryptor)
        print("✓ Successfully created encryption filter")
    except Exception as e:
        print(f"✗ Failed to create encryption filter: {e}")
        return False

    # Test creating decryption filter
    try:
        decrypt_filter = he_policy.create_decrypt_filter()
        assert isinstance(decrypt_filter, HEModelDecryptor)
        print("✓ Successfully created decryption filter")
    except Exception as e:
        print(f"✗ Failed to create decryption filter: {e}")
        return False

    # Test backward compatibility
    try:
        filter_instance = he_policy.create_filter()
        assert isinstance(filter_instance, HEModelEncryptor)
        print("✓ Backward compatibility works (create_filter returns encryption filter)")
    except Exception as e:
        print(f"✗ Backward compatibility failed: {e}")
        return False

    return True


def test_he_privacy_policy_variations():
    """Test different configurations of HEPrivacyPolicy."""
    print("\nTesting HEPrivacyPolicy variations...")

    # Test with layer-specific encryption
    he_policy_layers = HEPrivacyPolicy(
        tenseal_context_file="test_context.tenseal", encrypt_layers=["conv", "fc"], weigh_by_local_iter=True
    )

    try:
        encrypt_filter = he_policy_layers.create_encrypt_filter()
        assert isinstance(encrypt_filter, HEModelEncryptor)
        print("✓ Successfully created encryption filter with layer-specific encryption")
    except Exception as e:
        print(f"✗ Failed to create encryption filter with layer-specific encryption: {e}")
        return False

    # Test with regex pattern
    he_policy_regex = HEPrivacyPolicy(
        tenseal_context_file="test_context.tenseal", encrypt_layers="conv.*", weigh_by_local_iter=True
    )

    try:
        encrypt_filter = he_policy_regex.create_encrypt_filter()
        assert isinstance(encrypt_filter, HEModelEncryptor)
        print("✓ Successfully created encryption filter with regex pattern")
    except Exception as e:
        print(f"✗ Failed to create encryption filter with regex pattern: {e}")
        return False

    # Test with aggregation weights
    he_policy_weights = HEPrivacyPolicy(
        tenseal_context_file="test_context.tenseal",
        encrypt_layers=None,
        aggregation_weights={"site-1": 1.0, "site-2": 2.0},
        weigh_by_local_iter=True,
    )

    try:
        encrypt_filter = he_policy_weights.create_encrypt_filter()
        assert isinstance(encrypt_filter, HEModelEncryptor)
        print("✓ Successfully created encryption filter with aggregation weights")
    except Exception as e:
        print(f"✗ Failed to create encryption filter with aggregation weights: {e}")
        return False

    return True


def test_fedavg_recipe_with_he():
    """Test that FedAvgRecipe can be created with HE privacy policy."""
    print("\nTesting FedAvgRecipe with HE privacy policy...")

    try:
        # Create HE privacy policy
        he_policy = HEPrivacyPolicy(
            tenseal_context_file="test_context.tenseal", encrypt_layers=None, weigh_by_local_iter=True
        )

        # Create FedAvgRecipe with HE privacy policy
        recipe = FedAvgRecipe(train_script="test_script.py", num_clients=2, num_rounds=2, privacy_policies=[he_policy])

        print("✓ Successfully created FedAvgRecipe with HE privacy policy")
        print(f"✓ Number of privacy policies: {len(recipe.privacy_policies)}")

        # Check that the policy is in the list
        he_policies = [p for p in recipe.privacy_policies if isinstance(p, HEPrivacyPolicy)]
        assert len(he_policies) == 1
        print("✓ HE privacy policy is correctly included in the recipe")

        return True

    except Exception as e:
        print(f"✗ Failed to create FedAvgRecipe with HE privacy policy: {e}")
        return False


def main():
    """Run all tests."""
    print("Running HEPrivacyPolicy tests...\n")

    success = True

    # Run tests
    if not test_he_privacy_policy():
        success = False

    if not test_he_privacy_policy_variations():
        success = False

    if not test_fedavg_recipe_with_he():
        success = False

    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed! HEPrivacyPolicy is working correctly.")
    else:
        print("✗ Some tests failed. Please check the implementation.")

    return success


if __name__ == "__main__":
    main()
