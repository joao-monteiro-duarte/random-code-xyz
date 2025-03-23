#!/usr/bin/env python3
"""
Entry point for the crypto trading pool application.
This script launches either the FastAPI web application or runs a single processing cycle.
"""

import argparse
import asyncio
import logging
import sys
import os
import uvicorn

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('process_log.txt')
    ]
)

logger = logging.getLogger(__name__)

async def run_single_cycle():
    """Run a single processing cycle."""
    from config.settings import VPH_THRESHOLD
    from run_cycle_impl import run_cycle
    
    logger.info("Running a single processing cycle")
    await run_cycle(VPH_THRESHOLD)
    logger.info("Cycle completed")

def run_api_server(host='0.0.0.0', port=8000):
    """Run the FastAPI server."""
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run("api:app", host=host, port=port, reload=False, log_level="info")

def main():
    """Parse arguments and run the application."""
    parser = argparse.ArgumentParser(description="Crypto Trading Pool Application")
    parser.add_argument("--cycle", action="store_true", help="Run a single processing cycle")
    parser.add_argument("--api", action="store_true", help="Run the API server")
    parser.add_argument("--host", default="0.0.0.0", help="API server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="API server port (default: 8000)")
    
    args = parser.parse_args()
    
    # Initialize services
    try:
        # Import service architecture
        from services.app_service import AppService
        import redis
        from config.settings import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB
        
        # Try to connect to Redis
        try:
            redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD or None,
                db=REDIS_DB,
                decode_responses=True
            )
            redis_client.ping()
            redis_available = True
            logger.info("Connected to Redis successfully")
        except Exception as e:
            redis_client = None
            redis_available = False
            logger.warning(f"Redis connection failed: {e}")
        
        # Initialize app service (global instance)
        from services.app_service import app_service
        from config.settings import OPENROUTER_API_KEY
        
    except ImportError as e:
        logger.warning(f"Service architecture not available: {e}")
    
    # Run the specified mode
    if args.cycle:
        # Run a single cycle
        asyncio.run(run_single_cycle())
    elif args.api:
        # Run the API server
        run_api_server(host=args.host, port=args.port)
    else:
        # Default: Run the API server
        run_api_server()

if __name__ == "__main__":
    main()