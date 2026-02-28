"""
MerLin-based Quantum Reservoir for Q-volution 2026 (Quandela track).

This module implements a fixed, non-trainable photonic quantum reservoir using
the MerLin (merlinquantum) framework and Perceval under the hood.

Design goals
------------
- Respect hardware-inspired limits: up to 20 modes and 10 photons.
- No amplitude encoding or state injection (angle/phase encoding only).
- Provide a simple NumPy-friendly API similar to the previous Qiskit-based
  `QuantumReservoir.get_reservoir_states`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch

import merlin as ML


TensorLike = Union[np.ndarray, torch.Tensor]


@dataclass
class ReservoirConfig:
    """Configuration for the MerLin quantum reservoir."""

    n_modes: int = 8
    n_photons: int = 4
    depth: int = 3
    data_scaling_factor: float = 1.0
    input_size: Optional[int] = None
    device: Optional[Union[str, torch.device]] = None

    def __post_init__(self) -> None:
        if self.n_modes <= 0:
            raise ValueError("n_modes must be positive.")
        if self.n_modes > 20:
            raise ValueError("n_modes must not exceed 20 (hardware constraint).")
        if self.n_photons <= 0:
            raise ValueError("n_photons must be positive.")
        if self.n_photons > 10:
            raise ValueError("n_photons must not exceed 10 (hardware constraint).")


class QuantumReservoir:
    """
    Photonic Quantum Reservoir based on MerLin's QuantumLayer.

    The reservoir is a fixed, non-trainable quantum circuit that:
    - Encodes classical inputs via phase (angle) encoding on selected modes.
    - Evolves them through a random interferometer-style circuit.
    - Outputs per-mode photon number expectations as non-linear features.

    Public API
    ----------
    - get_reservoir_states(data): NumPy-based interface used by downstream models.
    - output_dim: dimensionality of the quantum feature vector.
    """

    def __init__(
        self,
        config: ReservoirConfig,
    ) -> None:
        self.config = config

        # Resolve device
        self.device = (
            torch.device(config.device)
            if isinstance(config.device, (str, torch.device))
            else torch.device("cpu")
        )

        # Will be set once we see the first batch of data (if not provided)
        self.input_size: Optional[int] = config.input_size

        # Underlying MerLin layer (lazily constructed once input_size is known)
        self._quantum_layer: Optional[ML.QuantumLayer] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_circuit_builder(self, input_size: int) -> ML.CircuitBuilder:
        """
        Construct a fixed photonic circuit using CircuitBuilder.

        Encoding strategy:
        - Use angle (phase) encoding via `add_angle_encoding`, which attaches
          input-driven phase shifters on a subset of modes.
        - Map each classical feature to one logical angle feature. Because
          angle encoding is limited by the number of modes, we only use
          `min(input_size, n_modes)` features; extra classical features are
          truncated at the reservoir interface.
        """
        n_modes = self.config.n_modes
        builder = ML.CircuitBuilder(n_modes=n_modes)

        # Decide how many classical features we can encode directly.
        n_features = min(input_size, n_modes)

        # Angle encoding on the first n_features modes.
        # scale=data_scaling_factor ensures that normalized inputs are mapped to
        # physically meaningful rotation angles.
        builder.add_angle_encoding(
            modes=list(range(n_features)),
            name="x",
            scale=self.config.data_scaling_factor,
            subset_combinations=False,
        )

        # Add several fixed interferometer-style entangling layers.
        # These are non-trainable and provide rich internal dynamics.
        for _ in range(self.config.depth):
            # Generic interferometer over all modes, fixed parameters
            builder.add_entangling_layer(modes=None, trainable=False, model="mzi")

        return builder

    def _build_quantum_layer(self, input_size: int) -> None:
        """
        Instantiate the MerLin QuantumLayer with angle encoding.

        This method MUST:
        - Use amplitude_encoding=False (the default).
        - Provide n_photons <= 10 and n_modes <= 20.
        - Use a Fock computation space and mode expectation measurement.
        """
        if self._quantum_layer is not None:
            return

        if input_size <= 0:
            raise ValueError("input_size must be positive.")

        # Limit input_size by the number of modes; truncate extra features.
        n_effective_features = min(input_size, self.config.n_modes)
        self.input_size = n_effective_features

        builder = self._build_circuit_builder(input_size=n_effective_features)

        measurement_strategy = ML.MeasurementStrategy.mode_expectations(
            ML.ComputationSpace.FOCK
        )

        # Build the QuantumLayer.
        # IMPORTANT: `amplitude_encoding` is left as False (default).
        self._quantum_layer = ML.QuantumLayer(
            input_size=n_effective_features,
            builder=builder,
            n_photons=self.config.n_photons,
            measurement_strategy=measurement_strategy,
            amplitude_encoding=False,
            device=self.device,
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def output_dim(self) -> int:
        """
        Dimensionality of the quantum feature vector.

        Returns
        -------
        int
            Number of features output by the reservoir (typically = n_modes).
        """
        if self._quantum_layer is None:
            raise RuntimeError(
                "QuantumLayer has not been built yet. "
                "Call get_reservoir_states() with some data first."
            )

        # QuantumLayer exposes output_size via its internal metadata.
        # We derive it from a dummy forward pass if needed.
        dummy = torch.zeros(1, self.input_size, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            out = self._quantum_layer(dummy)
        return int(out.shape[-1])

    def _prepare_input_array(self, data: TensorLike) -> np.ndarray:
        """
        Normalize input shapes to 2D NumPy array [n_samples, n_features].
        """
        if isinstance(data, torch.Tensor):
            x = data.detach().cpu().numpy()
        else:
            x = np.asarray(data)

        if x.ndim == 1:
            x = x.reshape(1, -1)
        elif x.ndim == 3:
            # [n_samples, lookback, n_features] -> flatten last two dims
            n, t, f = x.shape
            x = x.reshape(n, t * f)
        elif x.ndim != 2:
            raise ValueError(
                f"Unsupported input shape {x.shape}. "
                "Expected 1D, 2D, or 3D array."
            )

        return x.astype(np.float32, copy=False)

    def get_reservoir_states(self, data: TensorLike) -> np.ndarray:
        """
        Extract quantum reservoir features from classical input data.

        Parameters
        ----------
        data : np.ndarray or torch.Tensor
            Input data with shape:
            - [n_samples, n_features], or
            - [n_samples, lookback, n_features], or
            - [n_features] (single sample).

        Returns
        -------
        np.ndarray
            Quantum features with shape [n_samples, output_dim].
        """
        x = self._prepare_input_array(data)

        # Lazily build the QuantumLayer based on the observed input size
        if self._quantum_layer is None:
            self._build_quantum_layer(input_size=x.shape[1])

        assert self.input_size is not None

        # Truncate or pad features to match the configured input_size.
        if x.shape[1] > self.input_size:
            x_proc = x[:, : self.input_size]
        elif x.shape[1] < self.input_size:
            pad_width = self.input_size - x.shape[1]
            x_proc = np.pad(x, ((0, 0), (0, pad_width)), mode="constant")
        else:
            x_proc = x

        torch_x = torch.from_numpy(x_proc).to(self.device, dtype=torch.float32)

        # Reservoir is conceptually non-trainable; we disable gradients.
        with torch.no_grad():
            features = self._quantum_layer(torch_x)

        return features.detach().cpu().numpy().astype(np.float32)

