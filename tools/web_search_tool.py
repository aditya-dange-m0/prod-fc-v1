"""
Web Search Tool for LangGraph
Migrated from Agno to LangChain
"""

import os
import json
import dotenv
from langchain.tools import tool
from parallel import Parallel

dotenv.load_dotenv()


@tool
def search_web(query: str) -> str:
    """
    Search the web using Parallel AI Search API and return results.

    Args:
        query: Search query string (used as objective)

    Returns:
        JSON string with search results or error message
    """
    api_key = os.getenv("PARALLEL_API_KEY")
    if not api_key:
        return json.dumps(
            {
                "success": False,
                "error": "PARALLEL_API_KEY not found in environment variables",
            }
        )

    # Get configuration from environment variables with defaults
    max_results = int(os.getenv("PARALLEL_MAX_RESULTS", "10"))
    processor = os.getenv("PARALLEL_PROCESSOR", "base")

    try:
        client = Parallel(api_key=api_key)
        search = client.beta.search(objective=query, processor=processor)

        # Extract results and convert to our format
        results = []
        if hasattr(search, "results") and search.results:
            for result in search.results[:max_results]:  # Limit results
                if hasattr(result, "__dict__"):
                    # Convert result object to dict - handle new format with excerpts
                    excerpts = getattr(result, "excerpts", [])
                    content = (
                        " ".join(excerpts)
                        if excerpts
                        else getattr(result, "content", "")
                    )

                    result_dict = {
                        "title": getattr(result, "title", "Untitled"),
                        "content": content,
                        "url": getattr(result, "url", ""),
                        "excerpts": excerpts,
                    }
                else:
                    # If result is already a dict - handle new format with excerpts
                    excerpts = result.get("excerpts", [])
                    content = (
                        " ".join(excerpts) if excerpts else result.get("content", "")
                    )

                    result_dict = {
                        "title": result.get("title", "Untitled"),
                        "content": content,
                        "url": result.get("url", ""),
                        "excerpts": excerpts,
                    }

                results.append(result_dict)

        return json.dumps(
            {
                "success": True,
                "query": query,
                "results": results,
                "total_results": len(results),
                "processor_used": processor,
            }
        )

    except ImportError:
        return json.dumps(
            {
                "success": False,
                "error": "parallel library not available. Install with: pip install parallel",
            }
        )
    except Exception as e:
        return json.dumps(
            {
                "success": False,
                "error": f"Search request failed: {str(e)}. Check PARALLEL_API_KEY, PARALLEL_MAX_RESULTS (default: 10), PARALLEL_PROCESSOR (default: base) env vars.",
            }
        )


# Export for agent
SEARCH_TOOL = search_web
