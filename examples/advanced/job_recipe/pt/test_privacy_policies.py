#!/usr/bin/env python3
"""
Test script to demonstrate the new privacy policies system.
"""

from nvflare.app_common.filters.percentile_privacy import PercentilePrivacy
from nvflare.app_common.filters.svt_privacy import SVTPrivacy
from nvflare.recipes.fedavg import FedAvgRecipe, PercentilePrivacyPolicy, PrivacyPolicy, SVTPrivacyPolicy


def test_built_in_policies():
    """Test that built-in privacy policies work correctly."""
    print("Testing built-in privacy policies...")

    # Test SVT privacy policy
    svt_policy = SVTPrivacyPolicy(fraction=0.1, epsilon=0.1, noise_var=0.1, gamma=1e-5, tau=1e-6, replace=True)

    svt_filter = svt_policy.create_filter()
    assert isinstance(svt_filter, SVTPrivacy)
    print("✓ SVTPrivacyPolicy works correctly")

    # Test percentile privacy policy
    percentile_policy = PercentilePrivacyPolicy(percentile=10, gamma=0.01)

    percentile_filter = percentile_policy.create_filter()
    assert isinstance(percentile_filter, PercentilePrivacy)
    print("✓ PercentilePrivacyPolicy works correctly")


def test_custom_policy():
    """Test that custom privacy policies work correctly."""
    print("Testing custom privacy policy...")

    class TestPrivacyPolicy(PrivacyPolicy):
        def __init__(self, value: float = 0.5):
            self.value = value

        def create_filter(self):
            return PercentilePrivacy(percentile=20, gamma=self.value)

    custom_policy = TestPrivacyPolicy(value=0.03)
    custom_filter = custom_policy.create_filter()

    assert isinstance(custom_filter, PercentilePrivacy)
    assert custom_filter.percentile == 20
    assert custom_filter.gamma == 0.03
    print("✓ Custom privacy policy works correctly")


def test_recipe_with_policies():
    """Test that FedAvgRecipe works with privacy policies."""
    print("Testing FedAvgRecipe with privacy policies...")

    # Create a simple recipe with privacy policies
    privacy_policies = [SVTPrivacyPolicy(epsilon=0.2), PercentilePrivacyPolicy(percentile=15)]

    recipe = FedAvgRecipe(train_script="test_script.py", num_clients=2, num_rounds=1, privacy_policies=privacy_policies)

    assert len(recipe.privacy_policies) == 2
    assert isinstance(recipe.privacy_policies[0], SVTPrivacyPolicy)
    assert isinstance(recipe.privacy_policies[1], PercentilePrivacyPolicy)
    print("✓ FedAvgRecipe with privacy policies works correctly")


def test_empty_policies():
    """Test that empty privacy policies work correctly."""
    print("Testing empty privacy policies...")

    recipe = FedAvgRecipe(train_script="test_script.py", num_clients=2, num_rounds=1, privacy_policies=[])  # Empty list

    assert len(recipe.privacy_policies) == 0
    print("✓ Empty privacy policies work correctly")


if __name__ == "__main__":
    print("Running privacy policies tests...\n")

    try:
        test_built_in_policies()
        test_custom_policy()
        test_recipe_with_policies()
        test_empty_policies()

        print("\n🎉 All tests passed! The privacy policies system is working correctly.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
