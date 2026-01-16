"""XML-RPC data service for domain-sampled data."""

import random
from collections.abc import Iterable
from pathlib import Path
from xmlrpc.client import Binary
from xmlrpc.server import SimpleXMLRPCServer

from .domain_dataset import DomainDataset
from .drift_controller import DriftController


class DataService:
    """Minimal XML-RPC service for sampling data by domain.

    Attributes:
        dataset: DomainDataset used to fetch samples.
        drift: Optional drift controller for domain selection.
    """
    def __init__(
        self,
        meta_path: Path | str,
        buffer_size: int = 32,
        drift: DriftController | None = None,
    ) -> None:
        """Initialize the service with dataset metadata.

        Args:
            meta_path: Path to `datasets/<name>/_meta.json`.
            buffer_size: Samples buffered per domain.
            drift: Optional drift controller.
        """
        self.dataset = DomainDataset(meta_path, buffer_size=buffer_size)
        self.drift = drift
        self._server: SimpleXMLRPCServer | None = None

    def attach_server(self, server: SimpleXMLRPCServer) -> None:
        """Attach the XML-RPC server instance for shutdown control.

        Args:
            server: Active XML-RPC server instance.
        """
        self._server = server

    def get_domains(self) -> tuple[str, ...]:
        """Return available domains for RPC clients.

        Returns:
            Tuple of domain names.
        """
        return tuple(self.dataset.domains)

    def get_data(self, args: Iterable[int]) -> tuple[Binary, int] | None:
        """Return a single sample for the given client and time step.

        Args:
            args: Iterable containing (cid, time_step).

        Returns:
            Tuple of (Binary bytes, label_idx) or None if unavailable.
        """
        # args: (cid, time_step) in this minimal RPC contract.
        cid, time_step = list(args)[:2]
        domain = self._select_domain(cid, time_step)
        sample = self.dataset.get_one(domain)
        if sample is None:
            return None
        data, label_idx = sample
        return Binary(data), label_idx

    def stop(self) -> bool:
        """Shutdown the XML-RPC server if attached.

        Returns:
            True if shutdown was triggered, False otherwise.
        """
        # Allow remote shutdown via RPC.
        if self._server is None:
            return False
        self._server.shutdown()
        return True

    def _select_domain(self, cid: int, time_step: int) -> str:
        """Select a domain given the client and time step.

        Args:
            cid: Client identifier.
            time_step: Current time step.

        Returns:
            Selected domain name.
        """
        domains = self.dataset.domains
        if not domains:
            raise ValueError("no domains available")
        if self.drift is None:
            # Default to uniform domain sampling when no drift controller.
            return random.choice(domains)
        self.drift.update_time_step(time_step)
        domain_idx = self.drift.sample(cid)
        return domains[domain_idx % len(domains)]


def serve_forever(
    meta_path: Path | str,
    host: str = "0.0.0.0",
    port: int = 8000,
    buffer_size: int = 32,
    drift: DriftController | None = None,
) -> None:
    """Start an XML-RPC server and block forever.

    Args:
        meta_path: Path to `datasets/<name>/_meta.json`.
        host: Hostname or IP address to bind.
        port: TCP port to bind.
        buffer_size: Samples buffered per domain.
        drift: Optional drift controller.
    """
    # Minimal XML-RPC server bootstrap.
    server = SimpleXMLRPCServer((host, port), allow_none=True, logRequests=False)
    service = DataService(meta_path, buffer_size=buffer_size, drift=drift)
    service.attach_server(server)
    server.register_instance(service)
    server.serve_forever()
