# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
from pathlib import Path
import argparse
import re
import logging

# append alt_e2eshark dir to path to allow importing without explicit pythonpath management
TEST_DIR = Path(__file__).parent
sys.path.append(TEST_DIR)

from e2e_testing.framework import *

# import frontend test configs:
from e2e_testing.test_configs.onnxconfig import (
    OnnxTestConfig,
    REDUCE_TO_LINALG_PIPELINE,
)

# import backend
from e2e_testing.backends import SimpleIREEBackend


def get_tests(groups):
    """imports tests based on groups specification"""
    combinations = True if groups == "all" or groups == "combinations" else False
    models = True if groups == "all" or groups == "models" else False
    operators = True if groups == "all" or groups == "operators" else False

    from e2e_testing.registry import GLOBAL_TEST_LIST

    # importing the test generating files will register them to GLOBAL_TEST_LIST
    if combinations:
        from onnx_tests.combinations import model
    if models:
        from onnx_tests.models import model
    if operators:
        from onnx_tests.operators import model

    return GLOBAL_TEST_LIST


def main(args):
    """Sets up config and test list based on CL args, then runs the tests"""
    # setup config
    if args.framework != "onnx":
        raise NotImplementedError("only onnx frontend supported now")
    pipeline = REDUCE_TO_LINALG_PIPELINE if args.torchtolinalg else []
    config = OnnxTestConfig(
        str(TEST_DIR), SimpleIREEBackend(hal_target_backend=args.backend), pipeline
    )
    # get test list
    # TODO: allow for no-run/mark xfails

    test_list = get_tests(args.groups)
    if args.test_filter:
        test_list = [
            test for test in test_list if re.match(args.test_filter, test.unique_name)
        ]

    # TODO: logging setup

    # TODO: staging setup

    run_tests(
        test_list,
        config,
        args.rundirectory,
        args.cachedir,
        args.no_artifacts,
        args.verbose,
    )


def run_tests(test_list, config, dir_name, cache_dir_name, no_artifacts, verbose):
    """runs tests in test_list based on config"""
    # TODO: multi-process, argparse, setup exception handling and logging

    # set up a parent log directory to store results
    parent_log_dir = str(TEST_DIR) + "/" + dir_name + "/"
    if not os.path.exists(parent_log_dir):
        os.mkdir(parent_log_dir)
    # set up a parent cache directory to store results
    parent_cache_dir = cache_dir_name
    if not os.path.exists(parent_cache_dir):
        os.mkdir(parent_cache_dir)

    num_passes = 0

    for t in test_list:

        if verbose:
            print(f"running test {t.unique_name}...")

        # set log directory for the individual test
        log_dir = parent_log_dir + t.unique_name + "/"
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        # set cache directory for the individual test
        cache_dir = parent_cache_dir + t.unique_name + "/"

        try:
            # TODO: convert staging to an Enum and figure out how to specify staging from args

            # set up test
            stage = "model_info_setup"
            # build an instance of the test class
            inst = t.model_constructor(t.unique_name, log_dir, cache_dir)
            # generate inputs from the test instance
            inputs = inst.construct_inputs()
            inputs.save_to(log_dir + "input")

            # run native inference
            stage = "native_inference"
            golden_outputs_raw = inst.forward(inputs)
            golden_outputs_raw.save_to(log_dir + "golden_output")

            # generate mlir from the instance using the config
            stage = "mlir_import"
            artifact_save_to = None if no_artifacts else log_dir
            mlir_module, func_name = config.mlir_import(inst, save_to=artifact_save_to)

            # apply torch-mlir lowerings
            stage = "torch_mlir"
            mlir_module = config.apply_torch_mlir_passes(
                mlir_module, save_to=artifact_save_to
            )

            # compile mlir_module using config (calls backend compile)
            stage = "compilation"
            buffer = config.compile(mlir_module, save_to=artifact_save_to)

            # run inference with the compiled module
            stage = "compiled_inference"
            outputs_raw = config.run(buffer, inputs, func_name=func_name)
            outputs_raw.save_to(log_dir + "output")

            # apply model-specific post-processing:
            stage = "post-processing"
            golden_outputs = inst.apply_postprocessing(golden_outputs_raw)
            outputs = inst.apply_postprocessing(outputs_raw)
        except Exception as e:
            log_exception(e, log_dir, stage, t.unique_name, verbose)
            continue

        # store the results
        result = TestResult(
            name=t.unique_name, input=inputs, gold_output=golden_outputs, output=outputs
        )
        # log the results
        test_passed = log_result(result, log_dir, [1e-4, 1e-4])
        num_passes += int(test_passed)
        if verbose:
            to_print = "\tPASS" if test_passed else "\tFAILED (Numerics)"
            print(to_print)
        elif not test_passed:
            print(f"FAILED: {t.unique_name}")

    print("\nTest Summary:")
    print(f"\tPASSES: {num_passes}\n\tTOTAL: {len(test_list)}")
    print(f"results stored in {parent_log_dir}")


def log_result(result, log_dir, tol):
    summary = result_comparison(result, tol)
    num_match = 0
    num_total = 0
    for s in summary:
        num_match += s.sum().item()
        num_total += s.nelement()
    percent_correct = num_match / num_total
    with open(log_dir + "inference_comparison.log", "w+") as f:
        f.write(
            f"matching values with (rtol,atol) = {tol}: {num_match} of {num_total} = {percent_correct*100}%\n"
        )
        f.write(f"Test Result:\n{result}")
    return num_match == num_total


def log_exception(e: Exception, path: str, stage: str, name: str, verbose: bool):
    log_filename = path + stage + ".log"
    base_str = f"Failed test at stage {stage} with exception:\n{e}\n"
    with open(log_filename, "w") as f:
        f.write(base_str)
        if verbose:
            print(f"\tFAILED ({stage})")
            import traceback

            traceback.print_exception(e, file=f)
        else:
            print(f"FAILED: {name}")


def _get_argparse():
    msg = "The run.py script to run e2e shark tests"
    parser = argparse.ArgumentParser(prog="run.py", description=msg, epilog="")

    # test config related arguments:
    parser.add_argument(
        "-b",
        "--backend",
        choices=["llvm-cpu", "amd-aie", "rocm", "vulkan"],
        default="llvm-cpu",
        help="Target backend i.e. hardware to run on",
    )
    parser.add_argument(
        "-f",
        "--framework",
        choices=["pytorch", "onnx", "tensorflow"],
        default="onnx",
        help="Run tests for given framework(s)",
    )
    parser.add_argument(
        "--torchtolinalg",
        action="store_true",
        default=False,
        help="Have torch-mlir-opt to produce linalg instead of torch mlir and pass that to iree-compile",
    )
    parser.add_argument(
        "--torchmlirimport",
        choices=["compile", "fximport"],
        default="fximport",
        help="Use torch_mlir.torchscript.compile, or Fx importer",
    )

    # arguments for customizing test-stages:
    parser.add_argument(
        "--runfrom",
        choices=["model-run", "torch-mlir", "iree-compile"],
        default="model-run",
        help="Start from model-run, or torch MLIR, or IREE compiled artefact",
    )
    parser.add_argument(
        "--runupto",
        choices=["torch-mlir", "iree-compile", "inference"],
        default="inference",
        help="Run upto torch MLIR generation, IREE compilation, or inference",
    )

    # test-list filtering arguments:
    parser.add_argument(
        "-g",
        "--groups",
        choices=["operators", "combinations", "models"],
        default="all",
        help="Run given group of tests",
    )
    parser.add_argument(
        "-t",
        "--test-filter",
        help="Run given specific test(s) only",
    )
    parser.add_argument(
        "--testsfile",
        help="A file with lists of tests (starting with framework name) to run",
    )

    # test tolerance
    parser.add_argument(
        "--tolerance",
        help="Set abolsulte (atol) and relative (rtol) tolerances for comparing floating point numbers. Example: --tolerance 1e-03 1-04",
        nargs="+",
        type=float,
    )

    # logging
    parser.add_argument(
        "--cachedir",
        help="Please select a dir with large free space to cache all torch, hf, turbine_tank model data",
        required=True,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print aditional messsages to show progress",
    )
    parser.add_argument(
        "-r",
        "--rundirectory",
        default="test-run",
        help="The test run directory",
    )
    parser.add_argument(
        "--no-artifacts",
        help="If enabled, this flag will prevent saving mlir and vmfb files",
        action="store_true",
        default=False,
    )
    # parser.add_argument(
    #     "-d",
    #     "--todtype",
    #     choices=["default", "fp32", "fp16", "bf16"],
    #     default="default",
    #     help="If not default, casts model and input to given data type if framework supports model.to(dtype) and tensor.to(dtype)",
    # )
    # parser.add_argument(
    #     "-j",
    #     "--jobs",
    #     type=int,
    #     default=4,
    #     help="Number of parallel processes to use per machine for running tests",
    # )
    # parser.add_argument(
    #     "-m",
    #     "--mode",
    #     choices=["direct", "turbine", "onnx", "ort", "vaiml"],
    #     default="onnx",
    #     help="direct=Fx/TS->torch-mlir, turbine=aot-export->torch-mlir, onnx=exportonnx-to-torch-mlir, ort=exportonnx-to-ortep",
    # )
    # parser.add_argument(
    #     "--norun",
    #     action="store_true",
    #     default=False,
    #     help="Skip running of tests. Useful for generating test summary after the run",
    # )
    # parser.add_argument(
    #     "--report",
    #     action="store_true",
    #     default=False,
    #     help="Generate test report summary",
    # )
    # parser.add_argument(
    #     "--reportformat",
    #     choices=["pipe", "github", "html", "csv"],
    #     default="pipe",
    #     help="Format of the test report summary file. It takes subset of tablefmt value of python tabulate",
    # )
    # parser.add_argument(
    #     "--uploadtestsfile",
    #     help="A file with lists of tests that should be uploaded",
    # )
    return parser


if __name__ == "__main__":
    parser = _get_argparse()
    main(parser.parse_args())
