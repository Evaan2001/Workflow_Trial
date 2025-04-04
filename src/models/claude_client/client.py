from anthropic import Anthropic
import json
import os
from dotenv import load_dotenv


class ClaudeClient:
    def __init__(self, api_key=None):
        self.client = Anthropic(api_key=api_key)

    def get_structured_response(self, system_prompt, user_message, response_schema):
        """
        Get a structured response from Claude using the Model Context Protocol

        Args:
            system_prompt: Instructions for Claude
            user_message: The user's input
            response_schema: JSON schema defining the expected response format

        Returns:
            Structured response conforming to the schema
        """

        # Define JSON tool with the provided schema
        tools = [
            {
                "name": "structured_response",
                "description": "Returns a structured JSON response",
                "input_schema": response_schema,
            },
        ]

        response = self.client.messages.create(
            model="claude-3-haiku-20240307",
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=1024,
            temperature=0,
            tools=tools,
            tool_choice={"type": "tool", "name": "structured_response"},
        )

        # Extract the structured response from the tool calls
        if response.content:
            for content_block in response.content:
                if (
                    hasattr(content_block, "type")
                    and content_block.type == "tool_use"
                    and content_block.name == "structured_response"
                ):
                    return content_block.input

        # If no tool use was found, just return None with a print warning
        text_response = None
        if response.content:
            for content_block in response.content:
                if hasattr(content_block, "type") and content_block.type == "text":
                    text_response = content_block.text
                    break

        print(
            "WARNING: Claude didn't use the provided schema and responded with text instead."
        )
        if text_response:
            print(f"Claude's text response: {text_response[:100]}...")

        return None  # Return None to indicate no structured response


if __name__ == "__main__":

    # Load environment variables from .env file (just one line!)
    load_dotenv()
    # Get API key from environment
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    run_api_test = os.getenv("RUN_API_TESTS", "False").lower() == "true"

    print("Retrieved API key from environment:", anthropic_api_key)
    if not anthropic_api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found in environment variables. Please check your .env file."
        )

    if run_api_test == False:
        print(
            "Skipping tests that use Anthropic's API since RUN_API_TESTS is set to false in the .env file at the project's root."
        )
        exit()

    MyClaudeClient = ClaudeClient(api_key=anthropic_api_key)

    example_files = ["sentiment_example.json", "event_example.json"]
    for example_file in example_files:
        try:
            # Get the file path relative to the current script
            file_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), example_file
            )

            # Load example data
            with open(file_path, "r") as f:
                example_data = json.load(f)

            system_prompt = example_data["system_prompt"]
            user_message = example_data["user_message"]
            response_schema = example_data["response_schema"]

            print(f"\nProcessing example: {os.path.basename(example_file)}")

            # Get structured response
            result = MyClaudeClient.get_structured_response(
                system_prompt, user_message, response_schema
            )

            print(f"Result:")
            if result:
                print(json.dumps(result, indent=2))
            else:
                print("No structured response received")

        except FileNotFoundError:
            print(f"Example file not found: {example_file}")
        except json.JSONDecodeError:
            print(f"Invalid JSON in example file: {example_file}")
        except Exception as e:
            print(f"Error processing example {example_file}: {e}")

        print("-" * 50)
