#!/usr/bin/env python3
"""
Secure Aggregation Implementation

This module provides secure aggregation capabilities for federated learning,
including homomorphic encryption concepts, secure multi-party computation,
and privacy-preserving model updates.

Author: Senior Autonomous Engineering Agent
License: MIT
"""

import hashlib
import json
import numpy as np
import secrets
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class SecureUpdate:
    """Secure model update with metadata"""
    client_id: str
    update_hash: str
    encrypted_update: bytes
    signature: str
    timestamp: float
    nonce: str


class HomomorphicEncryption:
    """Simplified homomorphic encryption for demonstration"""
    
    def __init__(self, key_size: int = 1024):
        self.key_size = key_size
        # In a real implementation, this would use proper cryptographic libraries
        self.public_key = self._generate_public_key()
        self.private_key = self._generate_private_key()
    
    def _generate_public_key(self) -> int:
        """Generate a simple public key (for demonstration)"""
        return secrets.randbelow(2**self.key_size)
    
    def _generate_private_key(self) -> int:
        """Generate a simple private key (for demonstration)"""
        return secrets.randbelow(2**self.key_size)
    
    def encrypt(self, value: float) -> int:
        """Encrypt a single value"""
        # Simplified encryption: value + random noise
        noise = secrets.randbelow(1000)
        return int(value * 1000) + noise
    
    def decrypt(self, encrypted_value: int) -> float:
        """Decrypt a single value"""
        # Simplified decryption: remove noise and scale back
        return (encrypted_value % 1000) / 1000.0
    
    def add_encrypted(self, enc1: int, enc2: int) -> int:
        """Add two encrypted values homomorphically"""
        # Use a smaller modulus to avoid overflow
        modulus = min(2**32, 2**self.key_size)
        return (enc1 + enc2) % modulus
    
    def get_public_key(self) -> int:
        """Get public key for sharing"""
        return self.public_key


class SecureAggregator:
    """Secure aggregator for federated learning updates"""
    
    def __init__(self, num_clients: int, threshold: int = None):
        self.num_clients = num_clients
        self.threshold = threshold or (num_clients // 2 + 1)
        self.encryption = HomomorphicEncryption()
        self.client_keys = {}
        self.aggregation_buffer = {}
        
    def register_client(self, client_id: str, public_key: int) -> str:
        """Register a client and return session token"""
        session_token = secrets.token_hex(16)
        self.client_keys[client_id] = {
            'public_key': public_key,
            'session_token': session_token,
            'registered': True
        }
        return session_token
    
    def submit_update(
        self, 
        client_id: str, 
        session_token: str, 
        model_update: Dict[str, np.ndarray]
    ) -> bool:
        """Submit a secure model update"""
        if not self._verify_session(client_id, session_token):
            return False
        
        # Create secure update
        update_bytes = self._serialize_update(model_update)
        update_hash = hashlib.sha256(update_bytes).hexdigest()
        
        # Encrypt the update
        encrypted_update = self._encrypt_update(model_update)
        
        # Create signature (simplified)
        signature = self._create_signature(client_id, update_hash)
        
        secure_update = SecureUpdate(
            client_id=client_id,
            update_hash=update_hash,
            encrypted_update=encrypted_update,
            signature=signature,
            timestamp=secrets.token_hex(8),
            nonce=secrets.token_hex(16)
        )
        
        # Store in buffer
        self.aggregation_buffer[client_id] = secure_update
        return True
    
    def _verify_session(self, client_id: str, session_token: str) -> bool:
        """Verify client session token"""
        if client_id not in self.client_keys:
            return False
        return self.client_keys[client_id]['session_token'] == session_token
    
    def _serialize_update(self, model_update: Dict[str, np.ndarray]) -> bytes:
        """Serialize model update to bytes"""
        # Convert numpy arrays to lists for JSON serialization
        serializable_update = {}
        for key, value in model_update.items():
            if isinstance(value, np.ndarray):
                serializable_update[key] = value.tolist()
            else:
                serializable_update[key] = value
        
        return json.dumps(serializable_update, sort_keys=True).encode()
    
    def _encrypt_update(self, model_update: Dict[str, np.ndarray]) -> bytes:
        """Encrypt model update using homomorphic encryption"""
        encrypted_update = {}
        
        for key, value in model_update.items():
            if isinstance(value, np.ndarray):
                # Encrypt each element
                encrypted_array = np.vectorize(self.encryption.encrypt)(value)
                encrypted_update[key] = encrypted_array.tolist()
            else:
                encrypted_update[key] = self.encryption.encrypt(float(value))
        
        return json.dumps(encrypted_update).encode()
    
    def _create_signature(self, client_id: str, update_hash: str) -> str:
        """Create a simplified signature for the update"""
        # In a real implementation, this would use proper digital signatures
        message = f"{client_id}:{update_hash}"
        return hashlib.sha256(message.encode()).hexdigest()
    
    def aggregate_updates(self) -> Optional[Dict[str, np.ndarray]]:
        """Securely aggregate all submitted updates"""
        if len(self.aggregation_buffer) < self.threshold:
            return None
        
        # Collect all encrypted updates
        encrypted_updates = list(self.aggregation_buffer.values())
        
        # Homomorphically aggregate encrypted values
        aggregated_update = self._homomorphic_aggregate(encrypted_updates)
        
        # Decrypt the aggregated result
        final_update = self._decrypt_aggregated(aggregated_update)
        
        return final_update
    
    def _homomorphic_aggregate(
        self, 
        encrypted_updates: List[SecureUpdate]
    ) -> Dict[str, Any]:
        """Aggregate encrypted updates homomorphically"""
        if not encrypted_updates:
            return {}
        
        # Get the first update to determine structure
        first_update = json.loads(encrypted_updates[0].encrypted_update.decode())
        aggregated = {}
        
        # Initialize aggregated structure
        for key in first_update.keys():
            if isinstance(first_update[key], list):
                # Handle array updates
                shape = np.array(first_update[key]).shape
                aggregated[key] = np.zeros(shape, dtype=int)
            else:
                # Handle scalar updates
                aggregated[key] = 0
        
        # Aggregate all updates
        for secure_update in encrypted_updates:
            encrypted_data = json.loads(secure_update.encrypted_update.decode())
            
            for key, value in encrypted_data.items():
                if isinstance(value, list):
                    # Array aggregation
                    encrypted_array = np.array(value)
                    aggregated[key] = self.encryption.add_encrypted(
                        aggregated[key], encrypted_array
                    )
                else:
                    # Scalar aggregation
                    aggregated[key] = self.encryption.add_encrypted(
                        aggregated[key], value
                    )
        
        return aggregated
    
    def _decrypt_aggregated(self, aggregated_update: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Decrypt the aggregated update"""
        decrypted_update = {}
        
        for key, value in aggregated_update.items():
            if isinstance(value, (list, np.ndarray)):
                # Decrypt array
                decrypted_array = np.vectorize(self.encryption.decrypt)(value)
                decrypted_update[key] = decrypted_array
            else:
                # Decrypt scalar
                decrypted_update[key] = self.encryption.decrypt(value)
        
        return decrypted_update
    
    def get_aggregation_status(self) -> Dict[str, Any]:
        """Get current aggregation status"""
        return {
            'total_clients': self.num_clients,
            'threshold': self.threshold,
            'registered_clients': len(self.client_keys),
            'submitted_updates': len(self.aggregation_buffer),
            'ready_for_aggregation': len(self.aggregation_buffer) >= self.threshold,
            'client_ids': list(self.aggregation_buffer.keys())
        }
    
    def clear_buffer(self):
        """Clear the aggregation buffer"""
        self.aggregation_buffer.clear()


class FederatedLearningSecurity:
    """Security utilities for federated learning"""
    
    def __init__(self):
        self.secure_aggregator = None
    
    def setup_secure_aggregation(self, num_clients: int, threshold: int = None):
        """Setup secure aggregation for federated learning"""
        self.secure_aggregator = SecureAggregator(num_clients, threshold)
    
    def register_federated_client(self, client_id: str) -> str:
        """Register a client for federated learning"""
        if not self.secure_aggregator:
            raise RuntimeError("Secure aggregation not initialized")
        
        # Generate a public key for the client
        client_encryption = HomomorphicEncryption()
        public_key = client_encryption.get_public_key()
        
        return self.secure_aggregator.register_client(client_id, public_key)
    
    def submit_federated_update(
        self, 
        client_id: str, 
        session_token: str, 
        model_update: Dict[str, np.ndarray]
    ) -> bool:
        """Submit a federated learning update"""
        if not self.secure_aggregator:
            raise RuntimeError("Secure aggregation not initialized")
        
        return self.secure_aggregator.submit_update(
            client_id, session_token, model_update
        )
    
    def aggregate_federated_updates(self) -> Optional[Dict[str, np.ndarray]]:
        """Aggregate federated learning updates"""
        if not self.secure_aggregator:
            raise RuntimeError("Secure aggregation not initialized")
        
        return self.secure_aggregator.aggregate_updates()
    
    def get_federated_status(self) -> Dict[str, Any]:
        """Get federated learning security status"""
        if not self.secure_aggregator:
            return {'status': 'not_initialized'}
        
        return self.secure_aggregator.get_aggregation_status()


def create_secure_client_session(client_id: str, num_clients: int) -> Tuple[str, str]:
    """Create a secure client session for federated learning"""
    # Initialize security
    security = FederatedLearningSecurity()
    security.setup_secure_aggregation(num_clients)
    
    # Register client
    session_token = security.register_federated_client(client_id)
    
    return session_token, str(security.secure_aggregator.encryption.get_public_key())


def secure_model_update_aggregation(
    client_updates: List[Tuple[str, Dict[str, np.ndarray]]],
    threshold: int = None
) -> Optional[Dict[str, np.ndarray]]:
    """Securely aggregate model updates from multiple clients"""
    if not client_updates:
        return None
    
    num_clients = len(client_updates)
    if threshold is None:
        threshold = num_clients // 2 + 1
    
    # Setup secure aggregation
    security = FederatedLearningSecurity()
    security.setup_secure_aggregation(num_clients, threshold)
    
    # Submit all updates
    for client_id, model_update in client_updates:
        # Create session for each client
        session_token, _ = create_secure_client_session(client_id, num_clients)
        
        # Submit update
        success = security.submit_federated_update(client_id, session_token, model_update)
        if not success:
            print(f"Warning: Failed to submit update from client {client_id}")
    
    # Aggregate updates
    aggregated_update = security.aggregate_federated_updates()
    
    return aggregated_update


# Utility functions for testing and demonstration
def generate_test_model_update(input_dim: int = 20, hidden_dim: int = 64) -> Dict[str, np.ndarray]:
    """Generate a test model update for demonstration"""
    return {
        'layer1.weight': np.random.randn(hidden_dim, input_dim),
        'layer1.bias': np.random.randn(hidden_dim),
        'layer2.weight': np.random.randn(hidden_dim, hidden_dim),
        'layer2.bias': np.random.randn(hidden_dim),
        'layer3.weight': np.random.randn(3, hidden_dim),
        'layer3.bias': np.random.randn(3)
    }


def verify_secure_aggregation(
    original_updates: List[Dict[str, np.ndarray]],
    aggregated_update: Dict[str, np.ndarray]
) -> bool:
    """Verify that secure aggregation produces correct results"""
    if not original_updates or not aggregated_update:
        return False
    
    # Calculate expected aggregation (simple average)
    expected_aggregated = {}
    for key in original_updates[0].keys():
        expected_aggregated[key] = np.mean([update[key] for update in original_updates], axis=0)
    
    # Compare with secure aggregation result
    for key in expected_aggregated.keys():
        if not np.allclose(expected_aggregated[key], aggregated_update[key], atol=1e-3):
            return False
    
    return True
