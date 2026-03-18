"""Visual embedding providers for Hermes.

Providers in this package implement the ``VisualEmbeddingProvider`` protocol
defined in ``hermes.embedding_provider``.  Each provider is an optional
dependency — import errors are caught by the loader in
``get_visual_embedding_providers()`` and logged as warnings.

Available providers (loaded on demand):
- ``jepa``:  V-JEPA based visual embeddings (requires ``torch``)
- ``clip``:  CLIP based visual embeddings (requires ``torch``, ``open_clip``)
"""
