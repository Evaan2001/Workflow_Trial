load("@rules_python//python:py_binary.bzl", "py_binary")

py_binary(
    name="huggingface_client",
    srcs=["transformers_client.py"],
    main="transformers_client.py",
    deps=[
        "@pypi//torch",
        "@pypi//transformers",
        "@pypi//scipy",
    ],
    visibility=["//visibility:public"],
)

sh_test(
    name="hugging_face_test",
    srcs=["run_hugging_face.sh"],
    data=[
        ":huggingface_client",
    ],
)
