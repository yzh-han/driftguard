"""Drift controller for domain sampling."""

import random
from dataclasses import dataclass, field


@dataclass
class DriftController:
    """Drift controller that maintains client-domain state across time steps.

    Attributes:
        domains: Ordered list of domains used for sampling.
        seed: Optional seed for deterministic behavior.
        current_time_step: Current time step.
        client_domain_state: Mapping of client_id -> domain_probabilities.
    """
    domains: list[str]
    seed: int | None = None
    current_time_step: int = 0
    client_domain_state: dict[int, list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize the internal random generator."""
        self._rng = random.Random(self.seed)

    def update_time_step(self) -> None:
        """Update to the given time step and potentially trigger drift.

        Args:
            time_step: New time step to update to.
        """
        self.current_time_step += 1
        # Re-seed per time step for consistent but varying randomness
        self._rng.seed(hash((self.seed, self.current_time_step)))
        
        # Trigger drift logic here if needed
        self._maybe_trigger_drift()

    def get_client_domain(self, client_id: int) -> int:
        """Get the current domain index for a client based on probabilities.

        Args:
            client_id: Client identifier.

        Returns:
            Integer index into the domains list sampled from client's distribution.
        """
        # Initialize client if not seen before
        if client_id not in self.client_domain_state:
            self.client_domain_state[client_id] = self._initialize_client_domain(client_id)
        
        probabilities = self.client_domain_state[client_id]
        return self._rng.choices(range(len(self.domains)), weights=probabilities)[0]

    def get_client_probabilities(self, client_id: int) -> list[float]:
        """Get the current domain probabilities for a client.

        Args:
            client_id: Client identifier.

        Returns:
            List of probabilities for each domain.
        """
        if client_id not in self.client_domain_state:
            self.client_domain_state[client_id] = self._initialize_client_domain(client_id)
        
        return self.client_domain_state[client_id].copy()

    def set_client_domain(self, client_id: int, domain_index: int) -> None:
        """Set a client to a specific domain with probability 1.0.

        Args:
            client_id: Client identifier.
            domain_index: Index into the domains list.
        """
        if not (0 <= domain_index < len(self.domains)):
            raise ValueError(f"Domain index {domain_index} out of range [0, {len(self.domains)})")
        
        probabilities = [0.0] * len(self.domains)
        probabilities[domain_index] = 1.0
        self.client_domain_state[client_id] = probabilities

    def set_client_probabilities(self, client_id: int, probabilities: list[float]) -> None:
        """Set a client's domain probabilities.

        Args:
            client_id: Client identifier.
            probabilities: List of probabilities for each domain (must sum to 1.0).
        """
        if len(probabilities) != len(self.domains):
            raise ValueError(f"Expected {len(self.domains)} probabilities, got {len(probabilities)}")
        
        if abs(sum(probabilities) - 1.0) > 1e-6:
            raise ValueError(f"Probabilities must sum to 1.0, got {sum(probabilities)}")
        
        self.client_domain_state[client_id] = probabilities.copy()

    def _initialize_client_domain(self, client_id: int) -> list[float]:
        """Initialize domain probabilities for a new client.

        Args:
            client_id: Client identifier.

        Returns:
            Initial domain probabilities for this client.
        """
        # Use client_id as additional seed for deterministic assignment
        temp_rng = random.Random(hash((self.seed, client_id, "init")))
        domain_idx = temp_rng.randrange(len(self.domains))
        
        # Initialize with single domain (probability 1.0)
        probabilities = [0.0] * len(self.domains)
        probabilities[domain_idx] = 1.0
        return probabilities

    def _maybe_trigger_drift(self) -> None:
        """Trigger drift events based on current time step.
        
        Override this method to implement specific drift patterns.
        """
        # Default: no drift (clients keep their domains)
        pass

    def sample(self, cid: int) -> int:
        """Sample a domain index for backward compatibility.

        Args:
            cid: Client identifier.

        Returns:
            Integer index into the domains list.
        """
        return self.get_client_domain(cid)
