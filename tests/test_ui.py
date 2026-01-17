"""Tests for the Hermes Test UI endpoints.

These tests verify the /ui endpoint and static file serving functionality.
"""

import pytest
from fastapi.testclient import TestClient
from hermes.main import app

client = TestClient(app)


class TestUIEndpoint:
    """Tests for the /ui endpoint."""

    def test_ui_endpoint_returns_html(self):
        """GET /ui should return 200 with HTML content containing 'Hermes'."""
        response = client.get("/ui")

        # Should return 200 OK
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        # Content-Type should be text/html
        content_type = response.headers.get("content-type", "")
        assert "text/html" in content_type, (
            f"Expected content-type to contain 'text/html', got '{content_type}'"
        )

        # Response body should contain "Hermes"
        assert "Hermes" in response.text, (
            "Expected response to contain 'Hermes'"
        )


class TestStaticFiles:
    """Tests for static file serving."""

    def test_static_files_accessible(self):
        """Static files mount should exist and serve files from /static path.

        This test verifies that the static file mount is configured correctly.
        Once app.js is created, /static/js/app.js should be accessible.
        For now, we verify that the /static path is mounted by checking
        that requests to /static don't return 404 with 'Not Found' detail
        (which would indicate no route exists), but instead return a
        file-not-found style response from StaticFiles.
        """
        # Request a file that should exist once implemented
        response = client.get("/static/js/app.js")

        # If static mount doesn't exist, we get a 404 with {"detail": "Not Found"}
        # If static mount exists but file doesn't, we get a 404 with different response
        #
        # For TDD: We expect this to fail until:
        # 1. The /static mount is added to main.py
        # 2. The app.js file is created
        #
        # Once both are done, this should return 200
        assert response.status_code == 200, (
            f"Expected /static/js/app.js to be accessible (200), got {response.status_code}. "
            "Ensure StaticFiles is mounted at /static and app.js exists."
        )
