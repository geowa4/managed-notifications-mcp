"""MCP server for searching managed service notification logs."""

import json
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from fastmcp import FastMCP
from sentence_transformers import SentenceTransformer


class NotificationSearchServer:
    """MCP server for searching managed notification logs."""
    
    def __init__(self, db_path: Path, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the search server with ChromaDB and embedding model."""
        self.db_path = db_path
        self.model = SentenceTransformer(model_name)
        self.client = None
        self.collection = None
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize ChromaDB client and collection."""
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"ChromaDB database not found at {self.db_path}. "
                "Please run scripts/build_embeddings.py first."
            )
        
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.client.get_collection("managed_notifications")
    
    def search_notifications(
        self, 
        problem_statement: str, 
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant notification logs based on a problem statement.
        
        Args:
            problem_statement: Description of the issue to search for
            max_results: Maximum number of results to return
            
        Returns:
            List of matching notification documents with metadata
        """
        if not self.collection:
            raise RuntimeError("Database not initialized")
        
        # Create embedding for the problem statement
        query_embedding = self.model.encode([problem_statement])
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=max_results
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            # Parse the full JSON from metadata
            try:
                full_json = json.loads(results['metadatas'][0][i]['full_json'])
            except (json.JSONDecodeError, KeyError):
                full_json = {}
            
            # Parse variables from JSON string back to list
            variables_str = results['metadatas'][0][i].get('variables', '[]')
            try:
                variables = json.loads(variables_str) if variables_str else []
            except (json.JSONDecodeError, TypeError):
                variables = []
            
            result = {
                "id": results['ids'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results else 0.0,
                "similarity": 1 - (results['distances'][0][i] if 'distances' in results else 0.0),
                "file_path": results['metadatas'][0][i].get('file_path', ''),
                "folder": results['metadatas'][0][i].get('folder', ''),
                "severity": results['metadatas'][0][i].get('severity', 'Unknown'),
                "service_name": results['metadatas'][0][i].get('service_name', 'Unknown'),
                "log_type": results['metadatas'][0][i].get('log_type', 'Unknown'),
                "internal_only": results['metadatas'][0][i].get('internal_only', False),
                "variables": variables,
                "document_text": results['documents'][0][i],
                "notification": full_json
            }
            
            formatted_results.append(result)
        
        return formatted_results


# Initialize the FastMCP server
mcp = FastMCP("Managed Notifications Search")

# Initialize the search server
project_root = Path(__file__).parent
db_path = project_root / "chroma_db"
search_server = NotificationSearchServer(db_path)


@mcp.tool()
def search_service_logs(
    problem_statement: str, 
    max_results: int = 5
) -> List[Dict[str, Any]]:
    """
    Search managed service notification logs for issues matching a problem statement.
    
    This tool searches through a database of managed service notifications to find
    logs that are semantically similar to the provided problem description.
    
    IMPORTANT: Many notifications contain variable placeholders (e.g., ${TIME}, ${REASON}, 
    ${POD}, ${NAMESPACE}) that need to be replaced with actual values. When you find a 
    relevant notification that contains variables, you should:
    1. Present the notification to the user
    2. Ask the user to provide values for each variable listed in the "variables" field
    3. Help the user interpolate the variables into the notification text
    4. Print the service log using the exact JSON given using the interpolated values
    
    Args:
        problem_statement: Description of the issue or problem you're investigating
        max_results: Maximum number of matching notifications to return (default: 5)
    
    Returns:
        List of matching notification documents with metadata including:
        - notification: Full JSON notification data with variable placeholders
        - variables: List of variable names that need interpolation (e.g., ["TIME", "REASON"])
        - file_path: Path to the original notification file
        - folder: Category folder (hcp, osd, rosa, etc.)
        - severity: Notification severity level
        - similarity: Similarity score (0-1, higher is more similar)
        
    Example variables you might encounter:
        - ${TIME}: Timestamp when the issue occurred
        - ${REASON}: Specific reason for the failure
        - ${POD}: Name of the affected pod
        - ${NAMESPACE}: Kubernetes namespace
        - ${CLUSTER_ID}: Cluster identifier
        - ${NUM_OF_WORKERS}: Number of worker nodes
    """
    try:
        results = search_server.search_notifications(
            problem_statement=problem_statement,
            max_results=max_results
        )
        
        if not results:
            return [{
                "message": "No matching notifications found",
                "suggestion": "Try using different keywords or check if the database contains relevant notifications"
            }]
        
        return results
        
    except Exception as e:
        return [{
            "error": f"Search failed: {str(e)}",
            "suggestion": "Ensure the embeddings database has been built by running scripts/build_embeddings.py"
        }]


@mcp.tool()
def get_database_stats() -> Dict[str, Any]:
    """
    Get statistics about the notification database.
    
    Returns:
        Information about the number of notifications and categories available.
    """
    try:
        if not search_server.collection:
            return {"error": "Database not initialized"}
        
        # Get collection info
        collection_count = search_server.collection.count()
        
        # Get all metadata to analyze folders
        all_results = search_server.collection.get()
        folders = set()
        severities = set()
        service_names = set()
        
        for metadata in all_results['metadatas']:
            folders.add(metadata.get('folder', 'unknown'))
            severities.add(metadata.get('severity', 'unknown'))
            service_names.add(metadata.get('service_name', 'unknown'))
        
        return {
            "total_notifications": collection_count,
            "folders": sorted(list(folders)),
            "severities": sorted(list(severities)),
            "service_names": sorted(list(service_names)),
            "database_path": str(search_server.db_path)
        }
        
    except Exception as e:
        return {"error": f"Failed to get database stats: {str(e)}"}


def main():
    """Entry point for the MCP server."""
    import os
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    mcp.run(transport="streamable-http", host=host, port=port)


if __name__ == "__main__":
    main()
