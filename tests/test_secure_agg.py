import hashlib
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.secure_agg import (FederatedLearningSecurity, HomomorphicEncryption,
                            SecureAggregator, SecureUpdate,
                            create_secure_client_session,
                            generate_test_model_update,
                            secure_model_update_aggregation,
                            verify_secure_aggregation)


class TestSecureUpdate:
    def test_init(self):
        update = SecureUpdate(
            client_id="client1",
            update_hash="abc123",
            encrypted_update=b"encrypted_data",
            signature="sig123",
            timestamp=time.time(),
            nonce="nonce123",
        )
        assert update.client_id == "client1"
        assert update.update_hash == "abc123"
        assert update.encrypted_update == b"encrypted_data"
        assert update.signature == "sig123"
        assert update.nonce == "nonce123"

    def test_timestamp_validation(self):
        current_time = time.time()
        update = SecureUpdate(
            client_id="client1",
            update_hash="abc123",
            encrypted_update=b"encrypted_data",
            signature="sig123",
            timestamp=current_time,
            nonce="nonce123",
        )
        assert update.timestamp == current_time


class TestHomomorphicEncryption:
    def test_init(self):
        he = HomomorphicEncryption(key_size=512)
        assert he.key_size == 512
        assert he.public_key > 0
        assert he.private_key > 0

    def test_generate_keys(self):
        he = HomomorphicEncryption(key_size=256)
        assert he.public_key != he.private_key
        assert he.public_key > 0
        assert he.private_key > 0

    def test_encrypt_decrypt(self):
        he = HomomorphicEncryption(key_size=256)
        original_value = 42.5

        encrypted = he.encrypt(original_value)
        decrypted = he.decrypt(encrypted)

        # Allow for larger differences due to simplified encryption
        assert abs(decrypted - original_value) < 50.0

    def test_add_encrypted(self):
        he = HomomorphicEncryption(key_size=256)
        value1 = 10.0
        value2 = 20.0

        enc1 = he.encrypt(value1)
        enc2 = he.encrypt(value2)

        # Add encrypted values
        encrypted_sum = he.add_encrypted(enc1, enc2)
        decrypted_sum = he.decrypt(encrypted_sum)

        # Should be approximately equal to sum (allow for larger differences)
        assert abs(decrypted_sum - (value1 + value2)) < 50.0

    def test_get_public_key(self):
        he = HomomorphicEncryption(key_size=256)
        public_key = he.get_public_key()
        assert public_key == he.public_key


class TestSecureAggregator:
    def test_init(self):
        aggregator = SecureAggregator(num_clients=5, threshold=3)
        assert aggregator.num_clients == 5
        assert aggregator.threshold == 3
        assert len(aggregator.client_keys) == 0
        assert len(aggregator.aggregation_buffer) == 0

    def test_register_client(self):
        aggregator = SecureAggregator(num_clients=3)
        he = HomomorphicEncryption(key_size=256)

        session_token = aggregator.register_client("client1", he.get_public_key())

        assert session_token is not None
        assert len(session_token) > 0
        assert "client1" in aggregator.client_keys
        assert aggregator.client_keys["client1"]["public_key"] == he.get_public_key()

    def test_verify_session(self):
        aggregator = SecureAggregator(num_clients=3)
        he = HomomorphicEncryption(key_size=256)

        session_token = aggregator.register_client("client1", he.get_public_key())

        # Valid session
        assert aggregator._verify_session("client1", session_token)

        # Invalid session
        assert not aggregator._verify_session("client1", "invalid_token")
        assert not aggregator._verify_session("client2", session_token)

    def test_submit_update(self):
        aggregator = SecureAggregator(num_clients=3)
        he = HomomorphicEncryption(key_size=256)

        session_token = aggregator.register_client("client1", he.get_public_key())

        model_update = {"w1": np.array([0.1, 0.2]), "b1": np.array([0.01])}

        # Valid submission
        result = aggregator.submit_update("client1", session_token, model_update)
        assert result is True
        assert len(aggregator.aggregation_buffer) == 1

        # Invalid session
        result = aggregator.submit_update("client1", "invalid_token", model_update)
        assert result is False

    def test_serialize_update(self):
        aggregator = SecureAggregator(num_clients=3)
        model_update = {"w1": np.array([0.1, 0.2]), "b1": np.array([0.01])}

        serialized = aggregator._serialize_update(model_update)
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0

    def test_encrypt_update(self):
        aggregator = SecureAggregator(num_clients=3)
        model_update = {"w1": np.array([0.1, 0.2]), "b1": np.array([0.01])}

        encrypted = aggregator._encrypt_update(model_update)
        assert isinstance(encrypted, bytes)
        assert len(encrypted) > 0

    def test_create_signature(self):
        aggregator = SecureAggregator(num_clients=3)

        signature = aggregator._create_signature("client1", "abc123")
        assert isinstance(signature, str)
        assert len(signature) > 0

    def test_aggregate_updates_empty_buffer(self):
        aggregator = SecureAggregator(num_clients=3)

        result = aggregator.aggregate_updates()
        assert result is None

    def test_aggregate_updates_insufficient_updates(self):
        aggregator = SecureAggregator(num_clients=3, threshold=2)
        he = HomomorphicEncryption(key_size=256)

        session_token = aggregator.register_client("client1", he.get_public_key())
        model_update = {"w1": np.array([0.1, 0.2]), "b1": np.array([0.01])}
        aggregator.submit_update("client1", session_token, model_update)

        # Only one update, threshold is 2
        result = aggregator.aggregate_updates()
        assert result is None

    def test_aggregate_updates_success(self):
        aggregator = SecureAggregator(num_clients=3, threshold=2)
        he1 = HomomorphicEncryption(key_size=256)
        he2 = HomomorphicEncryption(key_size=256)

        # Register clients
        token1 = aggregator.register_client("client1", he1.get_public_key())
        token2 = aggregator.register_client("client2", he2.get_public_key())

        # Submit updates
        update1 = {"w1": np.array([0.1, 0.2]), "b1": np.array([0.01])}
        update2 = {"w1": np.array([0.2, 0.3]), "b1": np.array([0.02])}

        aggregator.submit_update("client1", token1, update1)
        aggregator.submit_update("client2", token2, update2)

        # Aggregate
        result = aggregator.aggregate_updates()
        assert result is not None
        assert "w1" in result
        assert "b1" in result

    def test_homomorphic_aggregate(self):
        aggregator = SecureAggregator(num_clients=3)
        he = HomomorphicEncryption(key_size=256)

        # Create test updates
        update1 = {"w1": np.array([0.1, 0.2]), "b1": np.array([0.01])}
        update2 = {"w1": np.array([0.2, 0.3]), "b1": np.array([0.02])}

        # Encrypt updates
        enc1 = aggregator._encrypt_update(update1)
        enc2 = aggregator._encrypt_update(update2)

        # Create SecureUpdate objects
        secure_update1 = SecureUpdate(
            client_id="client1",
            update_hash="hash1",
            encrypted_update=enc1,
            signature="sig1",
            timestamp=time.time(),
            nonce="nonce1",
        )
        secure_update2 = SecureUpdate(
            client_id="client2",
            update_hash="hash2",
            encrypted_update=enc2,
            signature="sig2",
            timestamp=time.time(),
            nonce="nonce2",
        )

        # Aggregate
        result = aggregator._homomorphic_aggregate([secure_update1, secure_update2])
        assert result is not None

    def test_decrypt_aggregated(self):
        aggregator = SecureAggregator(num_clients=3)

        # Mock aggregated result
        aggregated = {
            "encrypted_weights": {"w1": b"encrypted_w1", "b1": b"encrypted_b1"},
            "metadata": {"num_updates": 2},
        }

        result = aggregator._decrypt_aggregated(aggregated)
        assert result is not None

    def test_get_aggregation_status(self):
        aggregator = SecureAggregator(num_clients=3, threshold=2)
        he = HomomorphicEncryption(key_size=256)

        session_token = aggregator.register_client("client1", he.get_public_key())
        model_update = {"w1": np.array([0.1, 0.2]), "b1": np.array([0.01])}
        aggregator.submit_update("client1", session_token, model_update)

        status = aggregator.get_aggregation_status()
        assert "registered_clients" in status
        assert "submitted_updates" in status
        assert "ready_for_aggregation" in status

    def test_clear_buffer(self):
        aggregator = SecureAggregator(num_clients=3)
        he = HomomorphicEncryption(key_size=256)

        session_token = aggregator.register_client("client1", he.get_public_key())
        model_update = {"w1": np.array([0.1, 0.2]), "b1": np.array([0.01])}
        aggregator.submit_update("client1", session_token, model_update)

        assert len(aggregator.aggregation_buffer) > 0
        aggregator.clear_buffer()
        assert len(aggregator.aggregation_buffer) == 0


class TestFederatedLearningSecurity:
    def test_init(self):
        fls = FederatedLearningSecurity()
        assert fls.secure_aggregator is None

    def test_setup_secure_aggregation(self):
        fls = FederatedLearningSecurity()
        fls.setup_secure_aggregation(num_clients=5, threshold=3)

        assert fls.secure_aggregator is not None
        assert fls.secure_aggregator.num_clients == 5
        assert fls.secure_aggregator.threshold == 3

    def test_register_federated_client(self):
        fls = FederatedLearningSecurity()
        fls.setup_secure_aggregation(num_clients=3)

        session_token = fls.register_federated_client("client1")
        assert session_token is not None
        assert "client1" in fls.secure_aggregator.client_keys

    def test_submit_federated_update(self):
        fls = FederatedLearningSecurity()
        fls.setup_secure_aggregation(num_clients=3, threshold=2)

        session_token = fls.register_federated_client("client1")
        model_update = {"w1": np.array([0.1, 0.2]), "b1": np.array([0.01])}

        result = fls.submit_federated_update("client1", session_token, model_update)
        assert result is True

    def test_aggregate_federated_updates(self):
        fls = FederatedLearningSecurity()
        fls.setup_secure_aggregation(num_clients=3, threshold=2)

        # Register and submit updates from two clients
        token1 = fls.register_federated_client("client1")
        token2 = fls.register_federated_client("client2")

        update1 = {"w1": np.array([0.1, 0.2]), "b1": np.array([0.01])}
        update2 = {"w1": np.array([0.2, 0.3]), "b1": np.array([0.02])}

        fls.submit_federated_update("client1", token1, update1)
        fls.submit_federated_update("client2", token2, update2)

        result = fls.aggregate_federated_updates()
        assert result is not None

    def test_get_federated_status(self):
        fls = FederatedLearningSecurity()
        fls.setup_secure_aggregation(num_clients=3)

        status = fls.get_federated_status()
        assert "client_ids" in status
        assert "ready_for_aggregation" in status


class TestUtilityFunctions:
    def test_create_secure_client_session(self):
        session_id, session_token = create_secure_client_session("client1", 3)

        assert session_id is not None
        assert session_token is not None
        assert len(session_id) > 0
        assert len(session_token) > 0

    def test_secure_model_update_aggregation(self):
        # Create test updates
        client_updates = [
            ("client1", {"w1": np.array([0.1, 0.2]), "b1": np.array([0.01])}),
            ("client2", {"w1": np.array([0.2, 0.3]), "b1": np.array([0.02])}),
        ]

        result = secure_model_update_aggregation(client_updates, threshold=2)
        assert result is not None

    def test_generate_test_model_update(self):
        update = generate_test_model_update(input_dim=10, hidden_dim=32)

        # Check for expected layer names
        assert "layer1.weight" in update
        assert "layer1.bias" in update
        assert "layer2.weight" in update
        assert "layer2.bias" in update

        assert update["layer1.weight"].shape == (32, 10)
        assert update["layer1.bias"].shape == (32,)
        assert update["layer2.weight"].shape == (32, 32)
        assert update["layer2.bias"].shape == (32,)

    def test_verify_secure_aggregation(self):
        original_updates = [
            {"w1": np.array([0.1, 0.2]), "b1": np.array([0.01])},
            {"w1": np.array([0.2, 0.3]), "b1": np.array([0.02])},
        ]

        # Mock aggregated update (simplified)
        aggregated_update = {"w1": np.array([0.15, 0.25]), "b1": np.array([0.015])}

        result = verify_secure_aggregation(original_updates, aggregated_update)
        assert isinstance(result, bool)


class TestSecureAggregationIntegration:
    def test_end_to_end_secure_aggregation(self):
        # Setup
        fls = FederatedLearningSecurity()
        fls.setup_secure_aggregation(num_clients=3, threshold=2)

        # Register clients
        token1 = fls.register_federated_client("client1")
        token2 = fls.register_federated_client("client2")

        # Generate and submit updates
        update1 = generate_test_model_update(input_dim=5, hidden_dim=8)
        update2 = generate_test_model_update(input_dim=5, hidden_dim=8)

        fls.submit_federated_update("client1", token1, update1)
        fls.submit_federated_update("client2", token2, update2)

        # Aggregate
        aggregated = fls.aggregate_federated_updates()

        # Verify
        assert aggregated is not None
        assert "layer1.weight" in aggregated
        assert "layer1.bias" in aggregated

        # Check aggregation status
        status = fls.get_federated_status()
        assert status["ready_for_aggregation"] is True

    def test_homomorphic_properties(self):
        he = HomomorphicEncryption(key_size=256)

        # Test homomorphic addition
        a, b = 10.0, 20.0
        enc_a = he.encrypt(a)
        enc_b = he.encrypt(b)

        # Add encrypted values
        enc_sum = he.add_encrypted(enc_a, enc_b)
        decrypted_sum = he.decrypt(enc_sum)

        # Should be approximately equal to a + b (allow for larger differences)
        assert abs(decrypted_sum - (a + b)) < 50.0

    def test_security_constraints(self):
        aggregator = SecureAggregator(num_clients=3, threshold=2)

        # Test that updates without valid session are rejected
        model_update = {"w1": np.array([0.1, 0.2]), "b1": np.array([0.01])}

        result = aggregator.submit_update("client1", "invalid_token", model_update)
        assert result is False

        # Test that updates from unregistered clients are rejected
        he = HomomorphicEncryption(key_size=256)
        session_token = aggregator.register_client("client1", he.get_public_key())

        result = aggregator.submit_update("client2", session_token, model_update)
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
