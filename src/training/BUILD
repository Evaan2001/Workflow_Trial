load("@rules_python//python:py_binary.bzl", "py_binary")

py_binary(
    name="train_model",
    srcs=["train_model.py"],
    deps=[
        "@pypi//torch",
        "@pypi//transformers",
        "@pypi//datasets",
        "@pypi//accelerate",
        "@pypi//psutil",
    ],
    visibility=["//visibility:public"],
)

# New genrule to run the training and capture the outputs
# genrule(
#     name="train_and_capture_model",
#     srcs=[],
#     outs=[
#         "model_output/config.json",
#         "model_output/pytorch_model.bin",
#         "model_output/tokenizer.json",
#         "model_output/tokenizer_config.json",
#         "model_output/vocab.txt",
#         # Add any other files that get generated
#     ],
#     cmd="$(location :train_model) --output_dir=$(@D)/model_output",
#     tools=[":train_model"],
# )

sh_test(
    name="training_test",
    srcs=["run_training.sh"],
    data=[
        ":train_model",
    ],
    size="large",
    # size="enormous",
)
