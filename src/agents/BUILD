load("@rules_python//python:py_binary.bzl", "py_binary")

py_binary(
    name="hybrid_agent",
    srcs=["assistant.py"],
    main="assistant.py",
    deps=[
        "@pypi//python_dotenv",
        "//src/models/claude_client:claude_client",
        "//src/models/huggingface:huggingface_client",
    ],
    data=[
        "//:.env",  # Moved from deps to data
    ],
    imports=[
        ".",
        "../..",  # For imports starting with 'src'
    ],
    visibility=["//visibility:public"],
)

sh_test(
    name="hybrid_agent_test",
    srcs=["run_assistant.sh"],
    data=[
        ":hybrid_agent",
    ],
)
