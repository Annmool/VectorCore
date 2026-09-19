"""
Unit tests for Scalar and Product Quantization.
"""

import numpy as np
import pytest
from vectordb.core.quantization import ScalarQuantizer, ProductQuantizer


def test_scalar_quantization():
    np.random.seed(42)
    vectors = np.random.randn(100, 64).astype(np.float32)

    quantizer = ScalarQuantizer(symmetric=True)
    quantizer.fit(vectors)
    codes = quantizer.quantize(vectors)

    assert codes.dtype == np.int8
    assert codes.shape == (100, 64)

    recon = quantizer.dequantize(codes)
    mse = np.mean((vectors - recon) ** 2)
    # Int8 quantization error should be very small
    assert mse < 0.05


def test_product_quantization():
    np.random.seed(42)
    # 200 vectors with dim=32, M=4 subvectors (d_sub=8), K=16 centroids
    vectors = np.random.randn(200, 32).astype(np.float32)

    pq = ProductQuantizer(num_subvectors=4, num_clusters=16, max_iter=10)
    pq.fit(vectors)
    codes = pq.quantize(vectors)

    assert codes.shape == (200, 4)
    assert codes.dtype == np.uint8

    query = np.random.randn(32).astype(np.float32)
    adc_dists = pq.compute_distances_with_adc(query, codes)

    recon = pq.dequantize(codes)
    exact_recon_dists = np.sum((recon - query) ** 2, axis=-1)

    assert np.allclose(adc_dists, exact_recon_dists, atol=1e-3)
