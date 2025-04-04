import json
import os
import sys

# Add root directory to Python path for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Try both import styles
try:
    # Try absolute import first
    from src.models.claude_client.client import ClaudeClient
    from src.models.huggingface.transformers_client import HuggingFaceClient
except ImportError:
    # Fall back to relative import if absolute fails
    from ..models.claude_client.client import ClaudeClient
    from ..models.huggingface.transformers_client import HuggingFaceClient

# try:
#     # Try the local import path first
#     from src.models.huggingface.transformers_client import HuggingFaceClient
# except ModuleNotFoundError:
#     # Fall back to the remote import path
#     from ..models.huggingface.transformers_client import HuggingFaceClient


class HybridAgent:
    def __init__(
        self,
        anthropic_api_key,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.claude_client = ClaudeClient(api_key=anthropic_api_key)
        self.embedding_client = HuggingFaceClient(
            task="embeddings", model_name=embedding_model
        )

    def process_query(self, query, context_documents=None):
        """
        Process a user query using a combination of Claude and local models

        Args:
            query: User's question or request
            context_documents: Optional list of documents to use as context

        Returns:
            Response to the user's query
        """
        # If we have context documents, use the embedding model to rank them
        if context_documents:
            query_embedding = self.embedding_client([query])[0]  # Updated method call
            # Get embeddings for all documents
            doc_embeddings = self.embedding_client(
                context_documents
            )  # Updated method call

            # Calculate relevance scores (dot product)
            import numpy as np

            relevance_scores = np.dot(doc_embeddings, query_embedding)

            # Sort documents by relevance
            sorted_indices = np.argsort(relevance_scores)[::-1]
            top_docs = [context_documents[i] for i in sorted_indices[:3]]

            # Include top documents as context for Claude
            context_text = "\n\n".join(top_docs)
            augmented_query = f"Query: {query}\n\nRelevant Context:\n{context_text}"
        else:
            augmented_query = query

        # Define the response schema for Claude
        response_schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
                "sources": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["answer", "confidence"],
        }

        # Get structured response from Claude
        system_prompt = """
        You are a helpful AI assistant that provides accurate, concise answers.
        If the context includes relevant information, use it to inform your answer.
        Always indicate your confidence in your answer on a scale from 0 to 1.
        If you use information from the context, list the relevant sources.
        """

        response = self.claude_client.get_structured_response(
            system_prompt=system_prompt,
            user_message=augmented_query,
            response_schema=response_schema,
        )

        # Add error handling for None response
        if response is None:
            return f"I'm sorry, I wasn't able to process your query: '{query}'. Please try again with a different question."

        # Format the final response
        final_response = f"{response.get('answer', 'No answer provided')}"

        if response.get("sources"):
            final_response += f"\n\nSources: {', '.join(response['sources'])}"

        return final_response


if __name__ == "__main__":
    import os
    import sys
    import numpy as np
    from dotenv import load_dotenv

    load_dotenv()

    # Basic test function to organize tests
    def run_tests():
        print("Running HybridAgent tests...")

        # Test 1: Test initialization
        print("\nTest 1: Testing agent initialization...")
        try:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                print(
                    "Warning: ANTHROPIC_API_KEY not found in environment. Using placeholder for test."
                )
                api_key = "dummy_key_for_testing"
            print(
                f"ANTHROPIC_API_KEY exists: {'Yes: '+api_key if os.environ.get('ANTHROPIC_API_KEY') else 'No'}"
            )
            print(f"RUN_API_TESTS value: {os.environ.get('RUN_API_TESTS', 'Not set')}")

            # For the updated HuggingFaceClient, we need to specify task="embeddings"
            agent = HybridAgent(
                anthropic_api_key=api_key,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            )
            print("✅ Agent initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize agent: {str(e)}")
            return False

        # Test 2: Test embedding model functionality
        print("\nTest 2: Testing embedding functionality...")
        try:
            # Create a simple test for embeddings
            test_texts = ["This is a test sentence.", "Another sentence to embed."]
            # Use the new client's __call__ method which internally calls get_embeddings for task="embeddings"
            embeddings = agent.embedding_client(test_texts)

            # Verify shape and type
            assert isinstance(
                embeddings, np.ndarray
            ), "Embeddings should be a numpy array"
            assert embeddings.shape[0] == len(
                test_texts
            ), f"Expected {len(test_texts)} embeddings, got {embeddings.shape[0]}"
            assert (
                embeddings.shape[1] > 0
            ), "Embedding vectors should have non-zero dimensions"

            print(f"✅ Embedding model working correctly. Shape: {embeddings.shape}")
        except Exception as e:
            print(f"❌ Embedding test failed: {str(e)}")
            return False

        # Test 3: Mock test for Claude client
        print("\nTest 3: Testing Claude client functionality...")
        try:
            # This is a simplified test since we don't want to make actual API calls
            has_client = (
                hasattr(agent, "claude_client") and agent.claude_client is not None
            )
            assert has_client, "Claude client not properly initialized"

            # Check if the client has the expected method
            has_method = hasattr(agent.claude_client, "get_structured_response")
            assert has_method, "Claude client missing get_structured_response method"

            print("✅ Claude client properly configured")
        except Exception as e:
            print(f"❌ Claude client test failed: {str(e)}")
            return False

        # Test 4: Test a simple query without context
        # For test 4 in your assistant.py
        print("\nTest 4: Testing simple query processing (no context)...")
        try:
            if os.environ.get("RUN_API_TESTS", "").lower() == "true" and os.environ.get(
                "ANTHROPIC_API_KEY"
            ):
                print("Running API test with Claude...")
                query = "What is the capital of France?"

                try:
                    # Try to print the API key (first few chars) to verify it's loaded correctly
                    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
                    print(f"API key loaded (first 5 chars): {api_key[:5]}...")

                    # Print before making the call
                    print("Calling process_query...")

                    # Debug the structured response directly
                    print("Testing Claude client directly first...")
                    system_prompt = "You are a helpful AI assistant that provides accurate, concise answers."
                    response_schema = {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["answer", "confidence"],
                    }

                    structured_response = agent.claude_client.get_structured_response(
                        system_prompt=system_prompt,
                        user_message=query,
                        response_schema=response_schema,
                    )

                    print(f"Direct Claude response: {structured_response}")

                    # Now test the full process_query method
                    response = agent.process_query(query)
                    print(f"Full process_query response: {response}")

                    assert response and isinstance(
                        response, str
                    ), "Response should be a non-empty string"
                    print(
                        f"✅ Query processed successfully. Response: {response[:50]}..."
                    )
                except Exception as e:
                    print(f"Error details during API call: {str(e)}")
                    import traceback

                    traceback.print_exc()
                    raise
            else:
                print(
                    "⚠️ Skipping API test - set RUN_API_TESTS=true and valid ANTHROPIC_API_KEY to run"
                )
        except Exception as e:
            print(f"❌ Query processing test failed: {str(e)}")
            return False

        # Test 5: Test query with context
        print("\nTest 5: Testing query with context...")
        try:
            # Create some test context documents
            context_docs = [
                "Paris is the capital of France.",
                "Tokyo is the capital of Japan.",
                "Washington D.C. is the capital of the United States.",
            ]

            # Compare embeddings of a query with context documents
            query = "What is the capital of France?"
            query_embedding = agent.embedding_client([query])[0]  # Updated method call
            doc_embeddings = agent.embedding_client(context_docs)  # Updated method call

            # Calculate similarities (dot product)
            similarities = np.dot(doc_embeddings, query_embedding)

            # The most similar document should be the one about France
            most_similar_idx = np.argmax(similarities)
            assert (
                most_similar_idx == 0
            ), f"Expected document 0 to be most similar, got {most_similar_idx}"

            print("✅ Context ranking working correctly")

            # Full API test with context
            if os.environ.get("RUN_API_TESTS", "").lower() == "true" and os.environ.get(
                "ANTHROPIC_API_KEY"
            ):
                response = agent.process_query(query, context_docs)
                assert response and isinstance(
                    response, str
                ), "Response should be a non-empty string"
                print(
                    f"✅ Query with context processed successfully. Response: {response[:50]}..."
                )
            else:
                print(
                    "⚠️ Skipping API test with context - set RUN_API_TESTS=true to run"
                )
        except Exception as e:
            print(f"❌ Context query test failed: {str(e)}")
            return False

        print("\nAll tests completed successfully! ✅")
        return True

    # Run the tests
    success = run_tests()
    sys.exit(0 if success else 1)
