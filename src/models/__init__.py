"""Model architectures package for Shoplifting Detection."""

from src.models.cnn_rnn import ResNetBiGRUAttention
from src.models.cnn3d import VideoCNN3D

__all__ = ["ResNetBiGRUAttention", "VideoCNN3D"]
