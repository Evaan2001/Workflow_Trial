#!/usr/bin/env python3
import os
import sys
import argparse
from dotenv import load_dotenv

# Add root directory to Python path for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    # Try absolute import first
    from src.agents.assistant import HybridAgent
    from src.cache_manager import ModelCacheManager
except ImportError:
    # Fall back to relative import if absolute fails
    from ..agents.assistant import HybridAgent
    from ..cache_manager import ModelCacheManager


def run_examples(agent, cache_manager):
    """Run example queries with the HybridAgent."""
    print("\n✨ Running HybridAgent Examples ✨\n")

    # Example 1: Basic query
    example_query = "What is Model Context Protocol?"
    print(f"🔍 Example 1: Basic Query")
    print(f'🔎 Query: "{example_query}"')
    print("=" * 50)
    response = agent.process_query(example_query)
    print("\n" + "=" * 50)
    print("🤖 Agent Response:")
    print("=" * 50)
    print(response)
    print("\n" + "=" * 70)

    # Example 2: Context-enhanced query
    example_query = "How does Hugging Face integration work with AI systems?"
    print(f"\n\n📚 Example 2: Context-Enhanced Query")
    print(f'🔎 Query: "{example_query}"')

    # Load sample context
    script_dir = os.path.dirname(os.path.abspath(__file__))
    context_file = os.path.join(script_dir, "sample_context.txt")

    if os.path.exists(context_file):
        print(f"📝 Loading context from sample_context.txt")
        with open(context_file, "r") as f:
            context_documents = f.read().split("\n\n")
        print(f"📚 Loaded {len(context_documents)} context documents.")

        print("=" * 50)
        response = agent.process_query(example_query, context_documents)
        print("\n" + "=" * 50)
        print("🤖 Agent Response:")
        print("=" * 50)
        print(response)
    else:
        print(f"❌ Sample context file not found at: {context_file}")
        print("Running without context instead...")
        print("=" * 50)
        response = agent.process_query(example_query)
        print("\n" + "=" * 50)
        print("🤖 Agent Response:")
        print("=" * 50)
        print(response)

    print("\n🚀 Examples completed! Use --help to see all available options. 🚀\n")


# Create a custom help formatter to include examples
class CustomHelpFormatter(argparse.HelpFormatter):
    def __init__(self, prog):
        super().__init__(prog, max_help_position=40, width=100)

    def _format_usage(self, usage, actions, groups, prefix):
        usage = super()._format_usage(usage, actions, groups, prefix)
        example_text = """
✨ HybridAgent Example Usage ✨

🔍 Basic Query Example:
bazel run //src/demo:complete_agent -- --query "What is Model Context Protocol?"

📚 Context-Enhanced Query Example:
bazel run //src/demo:complete_agent -- --query "How does Hugging Face integration work?" --context_file src/demo/sample_context.txt

🧪 Run Example Queries:
bazel run //src/demo:complete_agent -- --examples

🚀 Happy querying! 🚀\n
"""
        return f"{usage}\n{example_text}"


def main():

    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run the hybrid AI agent that combines Claude and local models",
        formatter_class=CustomHelpFormatter,
    )

    # Create mutually exclusive group for the modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--examples",
        action="store_true",
        help="Run example queries to demonstrate the agent's capabilities",
    )
    mode_group.add_argument(
        "--query",
        type=str,
        metavar="QUERY",
        help="The user query/question to process via Claude & MCP",
    )

    # Create a separate argument group for query-related options
    query_options = parser.add_argument_group(
        "query options", "These options can only be used with --query"
    )
    query_options.add_argument(
        "--context_file", type=str, help="Optional file containing context documents"
    )
    query_options.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model to use for context ranking",
    )
    query_options.add_argument(
        "--run_api",
        action="store_true",
        help="Force API calls even if RUN_API_TESTS is not set to 'true'",
    )

    args = parser.parse_args()

    # Validate that NO additional flags are used with examples
    if args.examples:
        if (
            args.context_file is not None
            or args.embedding_model != "sentence-transformers/all-MiniLM-L6-v2"
            or args.run_api
        ):
            parser.error("When using --examples, no other flags can be specified")

    # If examples flag is set, run the example queries
    if args.examples:
        # Check if API key is available and if we should run API tests
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        run_api_tests = (
            os.environ.get("RUN_API_TESTS", "False").lower() == "true" or args.run_api
        )

        if not run_api_tests:
            print(
                "⚠️  Warning: RUN_API_TESTS environment variable is not set to 'true'."
            )
            print("❌ Stopping the test for example cases.")
            print(
                "✅ To enable API calls for the example cases, set RUN_API_TESTS=true in your .env file"
            )
            sys.exit(1)

        if not api_key:
            print("❌ Error: ANTHROPIC_API_KEY not found in environment variables.")
            print("👉 Please set it in your .env file or as an environment variable.")
            sys.exit(1)

        # Initialize the cache manager for examples
        cache_manager = ModelCacheManager()

        # Initialize the agent for examples
        print("🤖 Initializing the HybridAgent for examples...")
        agent = HybridAgent(
            anthropic_api_key=api_key,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )

        # Run the example queries
        run_examples(agent, cache_manager)
        return 0

    # Check if API key is available and if we should run API tests
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    run_api_tests = (
        os.environ.get("RUN_API_TESTS", "False").lower() == "true" or args.run_api
    )

    if not run_api_tests:
        print("⚠️  Warning: RUN_API_TESTS environment variable is not set to 'true'.")
        print("❌ API calls will be disabled unless you use the --run_api flag.")
        print("✅ To enable API calls, either:")
        print("  1. Set RUN_API_TESTS=true in your .env file, or")
        print("  2. Use the --run_api command-line flag")
        sys.exit(1)

    if not api_key:
        print("❌ Error: ANTHROPIC_API_KEY not found in environment variables.")
        print("👉 Please set it in your .env file or as an environment variable.")
        sys.exit(1)

    # Initialize the cache manager
    cache_manager = ModelCacheManager()

    # Check if embedding model is cached
    print(f"🔍 Checking if embedding model {args.embedding_model} is cached...")
    is_cached = cache_manager.is_cached_by_hugging_face(args.embedding_model)
    print(f"💾 Embedding model is cached: {is_cached}")

    # Initialize the agent
    print("🤖 Initializing the HybridAgent...")
    agent = HybridAgent(anthropic_api_key=api_key, embedding_model=args.embedding_model)

    # Load context documents if provided
    context_documents = None
    if args.context_file:
        print(f"📚 Loading context from {args.context_file}...")
        if not os.path.exists(args.context_file):
            print(f"❌ Error: Context file {args.context_file} not found.")
            sys.exit(1)

        with open(args.context_file, "r") as f:
            context_documents = f.read().split("\n\n")
        print(f"📝 Loaded {len(context_documents)} context documents.")

    # Process the query
    print(f"🔎 Processing query: {args.query}")
    response = agent.process_query(args.query, context_documents)

    print("\n" + "=" * 50)
    print("🤖 Agent Response:")
    print("=" * 50)
    print(response)

    return 0


if __name__ == "__main__":
    sys.exit(main())
