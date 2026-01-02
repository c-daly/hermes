"""Unit tests for Milvus client module."""

import pytest
from unittest.mock import Mock, patch
from hermes import milvus_client


def test_is_milvus_available():
    """Test that Milvus availability check works."""
    # This should return True since pymilvus is installed
    assert milvus_client.is_milvus_available()


def test_milvus_configuration():
    """Test that Milvus configuration getters and constants are defined."""
    # Lazy-loaded getters
    assert hasattr(milvus_client, "get_milvus_host")
    assert hasattr(milvus_client, "get_milvus_port")
    assert hasattr(milvus_client, "get_collection_name")
    # Direct constant
    assert hasattr(milvus_client, "EMBEDDING_DIMENSION")
    assert milvus_client.EMBEDDING_DIMENSION == 384
    # Getters return strings
    assert isinstance(milvus_client.get_milvus_host(), str)
    assert isinstance(milvus_client.get_milvus_port(), str)
    assert isinstance(milvus_client.get_collection_name(), str)


@patch("hermes.milvus_client.connections")
def test_connect_milvus_success(mock_connections):
    """Test successful Milvus connection."""
    # Reset global state
    milvus_client._milvus_connected = False

    mock_connections.connect.return_value = None
    result = milvus_client.connect_milvus()

    assert result
    assert milvus_client._milvus_connected
    mock_connections.connect.assert_called_once()


@patch("hermes.milvus_client.connections")
def test_connect_milvus_failure(mock_connections):
    """Test failed Milvus connection."""
    # Reset global state
    milvus_client._milvus_connected = False

    mock_connections.connect.side_effect = Exception("Connection failed")
    result = milvus_client.connect_milvus()

    assert not result
    assert not milvus_client._milvus_connected


@patch("hermes.milvus_client.connections")
@patch("hermes.milvus_client.utility")
@patch("hermes.milvus_client.Collection")
def test_ensure_collection_creates_new(mock_collection, mock_utility, mock_connections):
    """Test that ensure_collection creates a new collection if it doesn't exist."""
    # Setup
    milvus_client._milvus_connected = True
    mock_utility.has_collection.return_value = False
    mock_collection_instance = Mock()
    mock_collection.return_value = mock_collection_instance

    # Execute
    result = milvus_client.ensure_collection()

    # Verify
    assert result == mock_collection_instance
    mock_utility.has_collection.assert_called_once_with(
        milvus_client.get_collection_name()
    )
    mock_collection_instance.create_index.assert_called_once()


@pytest.mark.asyncio
@patch("hermes.milvus_client.ensure_collection")
async def test_persist_embedding_success(mock_ensure_collection):
    """Test successful embedding persistence."""
    # Setup
    milvus_client._milvus_connected = True
    mock_collection = Mock()
    mock_collection.insert = Mock()
    mock_collection.flush = Mock()
    mock_ensure_collection.return_value = mock_collection

    # Execute
    result = await milvus_client.persist_embedding(
        embedding_id="test-id",
        embedding=[0.1, 0.2, 0.3],
        model="test-model",
        text="test text",
    )

    # Verify
    assert result
    mock_collection.insert.assert_called_once()
    mock_collection.flush.assert_called_once()


@pytest.mark.asyncio
async def test_persist_embedding_milvus_not_connected():
    """Test that persist_embedding returns False when Milvus is not connected."""
    # Setup
    milvus_client._milvus_connected = False

    # Execute
    result = await milvus_client.persist_embedding(
        embedding_id="test-id",
        embedding=[0.1, 0.2, 0.3],
        model="test-model",
        text="test text",
    )

    # Verify
    assert not result


@patch("hermes.milvus_client.connect_milvus")
@patch("hermes.milvus_client.ensure_collection")
def test_initialize_milvus_success(mock_ensure_collection, mock_connect):
    """Test successful Milvus initialization."""
    # Setup
    mock_connect.return_value = True
    mock_ensure_collection.return_value = Mock()

    # Execute
    result = milvus_client.initialize_milvus()

    # Verify
    assert result
    mock_connect.assert_called_once()
    mock_ensure_collection.assert_called_once()


@patch("hermes.milvus_client.connect_milvus")
def test_initialize_milvus_connection_failure(mock_connect):
    """Test Milvus initialization when connection fails."""
    # Setup
    mock_connect.return_value = False

    # Execute
    result = milvus_client.initialize_milvus()

    # Verify
    assert not result
    mock_connect.assert_called_once()
