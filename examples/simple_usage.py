"""Simple example demonstrating how to use the Hermes API.

This example shows how to interact with each of the Hermes endpoints.
Make sure the Hermes server is running before executing this script:

    uvicorn hermes.main:app --host 0.0.0.0 --port 8080

Then run this script:

    python examples/simple_usage.py
"""

import requests
import io

API_BASE_URL = "http://localhost:8080"


def test_root():
    """Test the root endpoint."""
    print("Testing root endpoint...")
    response = requests.get(f"{API_BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_stt():
    """Test speech-to-text endpoint."""
    print("Testing STT endpoint...")

    # Create a dummy audio file
    audio_data = b"fake audio data"
    files = {"audio": ("test.wav", io.BytesIO(audio_data), "audio/wav")}

    response = requests.post(f"{API_BASE_URL}/stt", files=files)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_tts():
    """Test text-to-speech endpoint."""
    print("Testing TTS endpoint...")

    payload = {
        "text": "Hello, this is a test of the text-to-speech system.",
        "voice": "default",
        "language": "en-US",
    }

    response = requests.post(f"{API_BASE_URL}/tts", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type')}")
    print(f"Audio data length: {len(response.content)} bytes")
    print()


def test_simple_nlp():
    """Test simple NLP endpoint."""
    print("Testing Simple NLP endpoint...")

    payload = {
        "text": "The quick brown fox jumps over the lazy dog.",
        "operations": ["tokenize", "pos_tag", "lemmatize"],
    }

    response = requests.post(f"{API_BASE_URL}/simple_nlp", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_embed_text():
    """Test text embedding endpoint."""
    print("Testing Embed Text endpoint...")

    payload = {"text": "This is a sample sentence for embedding.", "model": "default"}

    response = requests.post(f"{API_BASE_URL}/embed_text", json=payload)
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Embedding dimension: {data['dimension']}")
    print(f"Model: {data['model']}")
    print(f"First 10 values: {data['embedding'][:10]}")
    print()


def test_llm():
    """Test LLM gateway endpoint."""
    print("Testing LLM endpoint...")

    payload = {
        "messages": [
            {"role": "system", "content": "You speak in short sentences."},
            {"role": "user", "content": "Say hello from the Hermes example."},
        ]
    }

    response = requests.post(f"{API_BASE_URL}/llm", json=payload)
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Provider: {data['provider']}")
    print(f"Response: {data['choices'][0]['message']['content']}")
    print()


def main():
    """Run all examples."""
    print("=" * 60)
    print("Hermes API Usage Examples")
    print("=" * 60)
    print()

    try:
        test_root()
        test_stt()
        test_tts()
        test_simple_nlp()
        test_embed_text()
        test_llm()

        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Hermes API.")
        print("Make sure the server is running:")
        print("    uvicorn hermes.main:app --host 0.0.0.0 --port 8080")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
