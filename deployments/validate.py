#!/usr/bin/env python3
"""
Deployment validation script for Hermes API.

This script validates that Hermes is properly deployed and responding to requests.
It can be used in CI/CD pipelines or as a manual deployment verification tool.

Usage:
    python deployments/validate.py [--url URL] [--timeout TIMEOUT] [--verbose]

Examples:
    # Validate local deployment
    python deployments/validate.py

    # Validate remote deployment
    python deployments/validate.py --url https://hermes.example.com

    # Increase timeout for slow connections
    python deployments/validate.py --timeout 60 --verbose
"""

import argparse
import sys
import time

try:
    import httpx
except ImportError:
    print("Error: httpx is required. Install with: pip install httpx")
    sys.exit(1)


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_status(message: str, status: str = "info") -> None:
    """Print formatted status message."""
    prefix = {
        "success": f"{Colors.GREEN}✓{Colors.ENDC}",
        "error": f"{Colors.RED}✗{Colors.ENDC}",
        "warning": f"{Colors.YELLOW}⚠{Colors.ENDC}",
        "info": f"{Colors.BLUE}ℹ{Colors.ENDC}",
    }
    print(f"{prefix.get(status, '')} {message}")


def validate_root_endpoint(base_url: str, timeout: float, verbose: bool) -> bool:
    """Validate the root endpoint returns expected data."""
    print_status("Testing root endpoint (GET /)...", "info")

    try:
        response = httpx.get(f"{base_url}/", timeout=timeout)
        response.raise_for_status()

        data = response.json()

        # Validate required fields
        required_fields = ["name", "version", "description", "endpoints"]
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            print_status(
                f"Root endpoint missing required fields: {missing_fields}", "error"
            )
            return False

        if verbose:
            print(f"  Name: {data['name']}")
            print(f"  Version: {data['version']}")
            print(f"  Description: {data['description']}")
            print(f"  Endpoints: {', '.join(data['endpoints'])}")

        print_status("Root endpoint OK", "success")
        return True

    except httpx.HTTPError as e:
        print_status(f"Root endpoint failed: {e}", "error")
        return False
    except Exception as e:
        print_status(f"Unexpected error: {e}", "error")
        return False


def validate_health_endpoint(base_url: str, timeout: float, verbose: bool) -> bool:
    """Validate the health endpoint returns expected data."""
    print_status("Testing health endpoint (GET /health)...", "info")

    try:
        response = httpx.get(f"{base_url}/health", timeout=timeout)
        response.raise_for_status()

        data = response.json()

        # Validate required fields
        required_fields = ["status", "version", "services"]
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            print_status(
                f"Health endpoint missing required fields: {missing_fields}", "error"
            )
            return False

        status = data.get("status", "unknown")
        services = data.get("services", {})

        if verbose:
            print(f"  Overall status: {status}")
            print(f"  Version: {data.get('version')}")
            print("  Services:")
            for service, service_status in services.items():
                print(f"    - {service}: {service_status}")

        # Check if any critical services are unavailable
        if status == "degraded":
            print_status(
                "Health check shows degraded status (ML dependencies may not be installed)",
                "warning",
            )
        else:
            print_status("Health endpoint OK", "success")

        return True

    except httpx.HTTPError as e:
        print_status(f"Health endpoint failed: {e}", "error")
        return False
    except Exception as e:
        print_status(f"Unexpected error: {e}", "error")
        return False


def validate_api_docs(base_url: str, timeout: float, verbose: bool) -> bool:
    """Validate that API documentation is accessible."""
    print_status("Testing API documentation (GET /docs)...", "info")

    try:
        response = httpx.get(f"{base_url}/docs", timeout=timeout)
        response.raise_for_status()

        if "swagger" in response.text.lower() or "openapi" in response.text.lower():
            print_status("API documentation OK", "success")
            return True
        else:
            print_status("API documentation may not be properly configured", "warning")
            return False

    except httpx.HTTPError as e:
        print_status(f"API documentation not accessible: {e}", "warning")
        return False
    except Exception as e:
        print_status(f"Unexpected error: {e}", "error")
        return False


def wait_for_service(base_url: str, timeout: float, max_attempts: int = 30) -> bool:
    """Wait for the service to become available."""
    print_status(f"Waiting for service at {base_url}...", "info")

    for attempt in range(1, max_attempts + 1):
        try:
            response = httpx.get(f"{base_url}/", timeout=timeout)
            if response.status_code == 200:
                print_status(f"Service is available (attempt {attempt})", "success")
                return True
        except httpx.ConnectError:
            pass
        except Exception:
            pass

        if attempt < max_attempts:
            time.sleep(2)

    print_status(
        f"Service did not become available after {max_attempts} attempts", "error"
    )
    return False


def main() -> int:
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description="Validate Hermes API deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8080",
        help="Base URL of the Hermes API (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for service to become available before validation",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    print(f"\n{Colors.BOLD}Hermes API Deployment Validation{Colors.ENDC}")
    print(f"Target: {args.url}")
    print(f"Timeout: {args.timeout}s")
    print()

    # Wait for service if requested
    if args.wait:
        if not wait_for_service(args.url, args.timeout):
            return 1
        print()

    # Run validations
    results = {
        "root": validate_root_endpoint(args.url, args.timeout, args.verbose),
        "health": validate_health_endpoint(args.url, args.timeout, args.verbose),
        "docs": validate_api_docs(args.url, args.timeout, args.verbose),
    }

    print()
    print(f"{Colors.BOLD}Validation Summary:{Colors.ENDC}")

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test, result in results.items():
        status = "PASS" if result else "FAIL"
        color = Colors.GREEN if result else Colors.RED
        print(f"  {test.ljust(20)}: {color}{status}{Colors.ENDC}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print_status("All validations passed!", "success")
        return 0
    else:
        print_status(
            f"{total - passed} validation(s) failed. Please check the deployment.",
            "error",
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
