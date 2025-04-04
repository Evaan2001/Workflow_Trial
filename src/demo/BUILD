load("@rules_python//python:py_binary.bzl", "py_binary")

py_binary(
    name="complete_agent",
    srcs=["complete_agent.py"],
    main="complete_agent.py",
    deps=[
        "@pypi//python_dotenv",
        "//src:cache_manager",
        "//src/agents:hybrid_agent",
        "//src/models/claude_client:claude_client",
        "//src/models/huggingface:huggingface_client",
    ],
    data=["//:.env", "sample_context.txt"],  # Moved from deps to data
    imports=[
        ".",
        "../..",  # For imports starting with 'src'
    ],
    visibility=["//visibility:public"],
)

sh_test(
    name="complete_agent_test",
    srcs=["run_complete_agent.sh"],
    data=[
        ":complete_agent",
    ],
)
