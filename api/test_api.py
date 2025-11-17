"""
Test Script for FastAPI
========================
Quick tests to verify API is working
"""

import requests
import json
from loguru import logger

# API base URL
BASE_URL = "http://localhost:8000"


def test_health():
    """Test health check endpoint"""
    logger.info("\n🧪 Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")

    if response.status_code == 200:
        data = response.json()
        logger.info(f"✅ Health check passed")
        logger.info(f"   Status: {data['status']}")
        logger.info(f"   Version: {data['version']}")
        logger.info(f"   Agents loaded: {data['agents_loaded']}")
    else:
        logger.error(f"❌ Health check failed: {response.status_code}")

    print(json.dumps(response.json(), indent=2))


def test_query(query: str, endpoint: str = "query"):
    """Test query endpoint"""
    logger.info(f"\n🧪 Testing /{endpoint} with: '{query}'")

    payload = {"query": query, "top_k": 5}
    response = requests.post(f"{BASE_URL}/{endpoint}", json=payload)

    if response.status_code == 200:
        data = response.json()
        logger.info(f"✅ Query successful")
        logger.info(f"   Agent: {data['agent']}")
        logger.info(f"   Sources: {data['num_sources']}")
        logger.info(f"   Time: {data['processing_time']:.2f}s")
        logger.info(f"   Answer: {data['answer'][:100]}...")
    else:
        logger.error(f"❌ Query failed: {response.status_code}")
        logger.error(f"   {response.text}")

    print(json.dumps(response.json(), indent=2))


def test_stats():
    """Test stats endpoint"""
    logger.info("\n🧪 Testing /stats endpoint...")
    response = requests.get(f"{BASE_URL}/stats")

    if response.status_code == 200:
        logger.info(f"✅ Stats retrieved")
    else:
        logger.error(f"❌ Stats failed: {response.status_code}")

    print(json.dumps(response.json(), indent=2))


def main():
    """Run all tests"""
    logger.info("🚀 Starting API Tests\n")
    logger.info("=" * 60)

    try:
        # Test 1: Health check
        test_health()

        # Test 2: RAG query
        test_query("What are gaming license requirements?", "query")

        # Test 3: RAG endpoint directly
        test_query("Tell me about parking regulations", "rag")

        # Test 4: Web search endpoint
        test_query("Recent AI developments", "web-search")

        # Test 5: Stats
        test_stats()

        logger.info("\n" + "=" * 60)
        logger.info("✅ All API tests completed!")

    except requests.exceptions.ConnectionError:
        logger.error("❌ Could not connect to API server")
        logger.error("   Make sure the server is running: python api/fastapi_app.py")
    except Exception as e:
        logger.error(f"❌ Test error: {e}")


if __name__ == "__main__":
    main()