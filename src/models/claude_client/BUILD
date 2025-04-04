load("@rules_python//python:py_binary.bzl", "py_binary")

py_binary(
    name="claude_client",
    srcs=["client.py"],
    main="client.py",
    data=[
        "sentiment_example.json",
        "event_example.json",
        "//:.env",  # Add your .env file here
    ],
    deps=[
        "@pypi//anthropic",
        "@pypi//python_dotenv",
    ],
    visibility=["//visibility:public"],
)

sh_test(
    name="claude_client_test",
    srcs=["run_claude_client.sh"],
    data=[
        ":claude_client",
    ],
)
