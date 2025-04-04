load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_test.bzl", "py_test")
load("@pypi//:requirements.bzl", "requirement")

py_binary(
    name="numpy_script",
    srcs=["use_numpy.py"],
    main="use_numpy.py",  # Explicitly specify the main file
    deps=[
        requirement("numpy"),
        # requirement("torch"),
        # requirement("triton"),
    ],
)

sh_test(
    name="numpy_test",
    srcs=["run_use_numpy.sh"],
    data=[
        ":numpy_script",
    ],
    # size = "small",
)
