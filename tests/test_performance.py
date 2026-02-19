"""Performance tests for Hermes API.

Tests cover:
- Embedding generation latency (p50, p95, p99)
- Milvus insertion throughput
- Concurrent request handling (load testing)
- Memory usage under load
- Connection pool efficiency
- Cache hit rates (if caching implemented)

Note: Requires pytest-benchmark for detailed performance metrics.
Install with: pip install pytest-benchmark
"""

import pytest
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi.testclient import TestClient
from hermes.main import app

# Check if ML dependencies are available
try:
    from hermes import services

    ML_AVAILABLE = services.SENTENCE_TRANSFORMERS_AVAILABLE
    NLP_AVAILABLE = services.SPACY_AVAILABLE
except ImportError:
    ML_AVAILABLE = False
    NLP_AVAILABLE = False

# Check if pytest-benchmark is available
try:
    import pytest_benchmark  # noqa: F401

    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False

# Check if Milvus is available
try:
    from pymilvus import connections

    MILVUS_AVAILABLE = True

    def check_milvus():
        try:
            connections.connect(
                alias="perf_test", host="localhost", port="17530", timeout=2
            )
            connections.disconnect("perf_test")
            return True
        except Exception:
            return False

    MILVUS_CONNECTED = check_milvus()
except ImportError:
    MILVUS_AVAILABLE = False
    MILVUS_CONNECTED = False

client = TestClient(app)


@pytest.mark.skipif(not ML_AVAILABLE, reason="ML dependencies not installed")
@pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
class TestEmbeddingLatency:
    """Test embedding generation latency."""

    def test_single_embedding_latency(self, benchmark):
        """Benchmark single embedding generation."""

        def generate_embedding():
            response = client.post(
                "/embed_text", json={"text": "Performance test sentence."}
            )
            return response

        result = benchmark(generate_embedding)
        assert result.status_code == 200

    def test_short_text_latency(self):
        """Test latency for short text (< 10 words)."""
        short_text = "Quick test"

        latencies = []
        for _ in range(10):
            start = time.time()
            response = client.post("/embed_text", json={"text": short_text})
            latency = (time.time() - start) * 1000  # Convert to ms
            latencies.append(latency)
            assert response.status_code == 200

        # Calculate percentiles
        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile

        print(f"\nShort text latency - P50: {p50:.2f}ms, P95: {p95:.2f}ms")

        # Reasonable thresholds (adjust based on hardware)
        assert p50 < 1000, f"P50 latency too high: {p50}ms"
        assert p95 < 2000, f"P95 latency too high: {p95}ms"

    def test_medium_text_latency(self):
        """Test latency for medium text (50-100 words)."""
        medium_text = "This is a medium length text. " * 20  # ~100 words

        latencies = []
        for _ in range(10):
            start = time.time()
            response = client.post("/embed_text", json={"text": medium_text})
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            assert response.status_code == 200

        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18]

        print(f"\nMedium text latency - P50: {p50:.2f}ms, P95: {p95:.2f}ms")

        assert p50 < 1500, f"P50 latency too high: {p50}ms"
        assert p95 < 3000, f"P95 latency too high: {p95}ms"

    def test_long_text_latency(self):
        """Test latency for long text (500+ words)."""
        long_text = "This is a sentence for testing. " * 100  # ~600 words

        latencies = []
        for _ in range(5):  # Fewer iterations for long text
            start = time.time()
            response = client.post("/embed_text", json={"text": long_text})
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            assert response.status_code == 200

        p50 = statistics.median(latencies)

        print(f"\nLong text latency - P50: {p50:.2f}ms")

        # Long text may take longer
        assert p50 < 5000, f"P50 latency too high: {p50}ms"


@pytest.mark.skipif(
    not ML_AVAILABLE or not MILVUS_CONNECTED, reason="Requires ML and Milvus"
)
class TestMilvusThroughput:
    """Test Milvus insertion and search throughput."""

    def test_embedding_insertion_throughput(self):
        """Test throughput of embedding insertions to Milvus."""
        num_embeddings = 50

        start = time.time()
        for i in range(num_embeddings):
            response = client.post(
                "/embed_text", json={"text": f"Throughput test sentence {i}."}
            )
            assert response.status_code == 200

        duration = time.time() - start
        throughput = num_embeddings / duration

        print(f"\nInsertion throughput: {throughput:.2f} embeddings/second")

        # Should handle at least 5 embeddings per second
        assert throughput > 5, f"Throughput too low: {throughput}"

    def test_batch_embedding_throughput(self):
        """Test throughput when generating embeddings in batches."""
        batch_size = 10
        num_batches = 5

        start = time.time()
        for batch_num in range(num_batches):
            for i in range(batch_size):
                response = client.post(
                    "/embed_text", json={"text": f"Batch {batch_num} item {i}"}
                )
                assert response.status_code == 200

        duration = time.time() - start
        total_embeddings = batch_size * num_batches
        throughput = total_embeddings / duration

        print(f"\nBatch throughput: {throughput:.2f} embeddings/second")

        assert throughput > 5


@pytest.mark.skipif(not ML_AVAILABLE, reason="ML dependencies not installed")
class TestConcurrentHandling:
    """Test concurrent request handling and load capacity."""

    def test_concurrent_embeddings(self):
        """Test handling concurrent embedding requests."""
        num_concurrent = 10

        def generate_embedding(i):
            start = time.time()
            response = client.post("/embed_text", json={"text": f"Concurrent test {i}"})
            latency = (time.time() - start) * 1000
            return response.status_code, latency

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [
                executor.submit(generate_embedding, i) for i in range(num_concurrent)
            ]
            results = [future.result() for future in as_completed(futures)]

        total_time = time.time() - start_time

        # All requests should succeed
        status_codes = [r[0] for r in results]
        assert all(s == 200 for s in status_codes)

        # Calculate average latency
        latencies = [r[1] for r in results]
        avg_latency = statistics.mean(latencies)

        print(f"\nConcurrent requests: {num_concurrent}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average latency: {avg_latency:.2f}ms")

        # Should complete in reasonable time
        assert total_time < 30, f"Total time too high: {total_time}s"

    def test_sustained_load(self):
        """Test handling sustained load over time."""
        duration_seconds = 10
        request_count = 0

        start_time = time.time()
        errors = 0

        while time.time() - start_time < duration_seconds:
            response = client.post(
                "/embed_text", json={"text": f"Sustained load test {request_count}"}
            )
            if response.status_code != 200:
                errors += 1
            request_count += 1
            time.sleep(0.1)  # 10 requests per second

        requests_per_second = request_count / duration_seconds
        error_rate = errors / request_count if request_count > 0 else 0

        print("\nSustained load test:")
        print(f"Duration: {duration_seconds}s")
        print(f"Requests: {request_count}")
        print(f"RPS: {requests_per_second:.2f}")
        print(f"Error rate: {error_rate * 100:.2f}%")

        # Should handle sustained load with low error rate
        assert error_rate < 0.05, f"Error rate too high: {error_rate * 100}%"

    def test_burst_load(self):
        """Test handling burst of requests."""
        burst_size = 20

        start = time.time()

        responses = []
        for i in range(burst_size):
            response = client.post("/embed_text", json={"text": f"Burst test {i}"})
            responses.append(response)

        duration = time.time() - start

        success_count = sum(1 for r in responses if r.status_code == 200)
        success_rate = success_count / burst_size

        print("\nBurst load test:")
        print(f"Burst size: {burst_size}")
        print(f"Duration: {duration:.2f}s")
        print(f"Success rate: {success_rate * 100:.2f}%")

        # Most requests should succeed
        assert success_rate > 0.9, f"Success rate too low: {success_rate * 100}%"


@pytest.mark.skipif(not NLP_AVAILABLE, reason="NLP dependencies not installed")
@pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
class TestNLPPerformance:
    """Test NLP operation performance."""

    def test_tokenization_latency(self, benchmark):
        """Benchmark tokenization performance."""

        def tokenize():
            response = client.post(
                "/simple_nlp",
                json={"text": "This is a test sentence.", "operations": ["tokenize"]},
            )
            return response

        result = benchmark(tokenize)
        assert result.status_code == 200

    def test_nlp_multi_operation_latency(self):
        """Test latency for multiple NLP operations."""
        text = "The quick brown fox jumps over the lazy dog."
        operations = ["tokenize", "pos_tag", "lemmatize", "ner"]

        latencies = []
        for _ in range(10):
            start = time.time()
            response = client.post(
                "/simple_nlp", json={"text": text, "operations": operations}
            )
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            assert response.status_code == 200

        p50 = statistics.median(latencies)

        print(f"\nNLP multi-operation latency - P50: {p50:.2f}ms")

        assert p50 < 2000, f"P50 latency too high: {p50}ms"

    def test_concurrent_nlp_requests(self):
        """Test concurrent NLP processing."""
        num_concurrent = 5

        def process_nlp(i):
            start = time.time()
            response = client.post(
                "/simple_nlp",
                json={
                    "text": f"NLP test sentence number {i}.",
                    "operations": ["tokenize", "pos_tag"],
                },
            )
            latency = (time.time() - start) * 1000
            return response.status_code, latency

        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(process_nlp, i) for i in range(num_concurrent)]
            results = [future.result() for future in as_completed(futures)]

        status_codes = [r[0] for r in results]
        assert all(s == 200 for s in status_codes)


@pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
class TestAPIOverhead:
    """Test API overhead and response times."""

    def test_health_endpoint_latency(self, benchmark):
        """Benchmark health check endpoint."""

        def health_check():
            return client.get("/health")

        result = benchmark(health_check)
        assert result.status_code == 200

    def test_root_endpoint_latency(self, benchmark):
        """Benchmark root endpoint."""

        def root_request():
            return client.get("/")

        result = benchmark(root_request)
        assert result.status_code == 200

    def test_validation_error_overhead(self):
        """Test overhead of validation errors."""
        latencies = []

        for _ in range(20):
            start = time.time()
            response = client.post("/embed_text", json={"text": ""})
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            assert response.status_code == 400

        avg_latency = statistics.mean(latencies)

        print(f"\nValidation error overhead: {avg_latency:.2f}ms")

        # Validation should be fast
        assert avg_latency < 100, f"Validation overhead too high: {avg_latency}ms"


@pytest.mark.skipif(not ML_AVAILABLE, reason="ML dependencies not installed")
class TestCacheEfficiency:
    """Test caching behavior and efficiency (if implemented)."""

    def test_repeated_request_performance(self):
        """Test if repeated identical requests benefit from caching."""
        text = "Cache test sentence"

        # First request (cold)
        start = time.time()
        response1 = client.post("/embed_text", json={"text": text})
        cold_latency = (time.time() - start) * 1000
        assert response1.status_code == 200

        # Repeated requests (potentially cached)
        warm_latencies = []
        for _ in range(5):
            start = time.time()
            response = client.post("/embed_text", json={"text": text})
            warm_latency = (time.time() - start) * 1000
            warm_latencies.append(warm_latency)
            assert response.status_code == 200

        avg_warm_latency = statistics.mean(warm_latencies)

        print(f"\nCold latency: {cold_latency:.2f}ms")
        print(f"Warm latency: {avg_warm_latency:.2f}ms")

        # Note: Current implementation doesn't cache, so latencies will be similar
        # This test documents baseline behavior


@pytest.mark.skipif(not ML_AVAILABLE, reason="ML dependencies not installed")
class TestMemoryUsage:
    """Test memory usage patterns (requires memory_profiler)."""

    def test_embedding_memory_footprint(self):
        """Test memory usage for embedding generation."""
        # Generate multiple embeddings to observe memory pattern
        initial_embeddings = 10

        for i in range(initial_embeddings):
            response = client.post("/embed_text", json={"text": f"Memory test {i}"})
            assert response.status_code == 200

        # Note: Actual memory profiling requires additional tools
        # This test ensures the endpoint works under repeated calls
        # For detailed profiling, use memory_profiler or similar tools

    def test_large_batch_memory(self):
        """Test memory usage with large batch processing."""
        batch_size = 20

        for i in range(batch_size):
            response = client.post(
                "/embed_text", json={"text": f"Large batch item {i}" * 10}
            )
            assert response.status_code == 200


class TestPerformanceBaselines:
    """Establish performance baselines for monitoring."""

    def test_baseline_health_check(self):
        """Establish baseline for health check performance."""
        latencies = []

        for _ in range(50):
            start = time.time()
            response = client.get("/health")
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            assert response.status_code == 200

        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18]
        p99 = statistics.quantiles(latencies, n=100)[98]

        print("\nHealth check baseline:")
        print(f"P50: {p50:.2f}ms")
        print(f"P95: {p95:.2f}ms")
        print(f"P99: {p99:.2f}ms")

        # Health check should be very fast
        assert p50 < 50
        assert p95 < 100
        assert p99 < 200

    @pytest.mark.skipif(not ML_AVAILABLE, reason="ML dependencies not installed")
    def test_baseline_embedding(self):
        """Establish baseline for embedding generation."""
        latencies = []

        for i in range(30):
            start = time.time()
            response = client.post("/embed_text", json={"text": f"Baseline test {i}"})
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            assert response.status_code == 200

        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18]
        p99 = max(latencies)  # For small sample

        print("\nEmbedding generation baseline:")
        print(f"P50: {p50:.2f}ms")
        print(f"P95: {p95:.2f}ms")
        print(f"P99: {p99:.2f}ms")

        # Document baselines (thresholds depend on hardware)
        print("\nBaseline established - adjust thresholds based on your hardware")
