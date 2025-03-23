#!/usr/bin/env python3
"""
Script to run all tests for the crypto trading pool application using pytest.
"""

import sys
import os
import logging
import pytest

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def run_tests_with_pytest():
    """Run all tests using pytest."""
    logger.info("Running tests using pytest...")
    
    # Check if specific tests were requested
    if len(sys.argv) > 1:
        pytest_args = sys.argv[1:]
        logger.info(f"Running specific tests: {pytest_args}")
        return pytest.main(pytest_args)
    
    # Default behavior: run API tests first, then all other tests
    logger.info("Running API endpoint tests...")
    api_result = pytest.main([
        'tests/test_api_endpoints.py',
        '-v',
        '--asyncio-mode=auto'
    ])
    
    if api_result != 0:
        logger.warning("API endpoint tests failed. Check for errors.")
    
    logger.info("Running all other tests...")
    return pytest.main([
        'tests/',
        '--ignore=tests/test_api_endpoints.py',  # Skip the API tests we already ran
        '-v',
        '--asyncio-mode=auto'
    ])

if __name__ == "__main__":
    logger.info("Running tests for crypto trading pool application")
    
    # Run tests with pytest
    exit_code = run_tests_with_pytest()
    
    # Exit with appropriate code
    sys.exit(exit_code)