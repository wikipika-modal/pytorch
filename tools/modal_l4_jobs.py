#!/usr/bin/env python3
"""Run the L4 test jobs of a trunk commit in Modal sandboxes, as CI specifies them.

    python tools/modal_l4_jobs.py SHA [--only default:3 inductor:1] [--run-id ID]

SHA is a commit that trunk.yml has built. Each L4 job of
linux-jammy-cuda13.2-py3.11-gcc11 / test (default 1-14 and inductor 1-2) gets a
sandbox of its own: CI's image for the commit, the build artifact and the target
determination results that the trunk run published, one L4 with 11 CPUs and
41 GiB, and .ci/pytorch/test.sh with the job's environment. A job's log, exit
code and test reports go to the Modal Volume pytorch-l4-jobs, under
SHA/CONFIG-SHARD. Exits 1 if a job fails.

Needs `pip install modal` and `modal token new`.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import modal


REPO = "pytorch/pytorch"
BUILD_ENVIRONMENT = "linux-jammy-cuda13.2-py3.11-gcc11"
# CI tags the image with the hash of the commit's .ci/docker tree
CI_IMAGE = "ghcr.io/pytorch/ci-image:pytorch-linux-jammy-cuda13.2-cudnn9-py3-gcc11"
ARTIFACTS = "https://gha-artifacts.s3.amazonaws.com/pytorch/pytorch"
# trunk.yml's L4 entries for this build: each test config with its number of shards
MATRIX = {"default": 14, "inductor": 2}
# trunk.yml's tests-to-exclude for the job, which test.sh applies to the default config
TESTS_TO_EXCLUDE = (
    "inductor/test_torchinductor inductor/test_torchinductor_opinfo "
    "inductor/test_aot_inductor inductor/test_cpu_select_algorithm"
)
# The runner l-x86aavx2-11-41-l4, and the job's timeout-minutes. Modal takes the
# CPUs and memory as a request: a sandbox may use more
GPU = "L4"
CPUS = 11
MEMORY_MIB = 41 * 1024
TIMEOUT_SECONDS = 360 * 60
NAME = "pytorch-l4-jobs"  # the Modal app and Volume
WORKSPACE = "/__w/pytorch/pytorch"  # where CI runs the job
RUNNER_TEMP = "/tmp/runner"
# Tests that fail in Modal's sandboxes, which run on gVisor
DESELECTED = [
    "test/distributed/fsdp/test_utils.py::TestUtilsCUDA::test_get_param_to_fqns_scales_linearly_cuda",
    "test/test_cuda.py::TestExecuteOnStreamsCUDA::test_capture_replay_stream_kind_green_cuda",
    "test/test_cuda.py::TestExecuteOnStreamsCUDA::test_capture_replay_stream_kind_mixed_cuda",
]
# What the image holds on top of CI's: the checkout, the build and the wheel, as
# the job's setup steps leave them
SETUP = [
    "mkdir -p {workspace}",
    "cd {workspace}",
    "git init -q .",
    "git remote add origin https://github.com/{repo}",
    "git fetch -q --depth 1 origin {sha}",
    "git checkout -q FETCH_HEAD",
    "curl -fsSL -o artifacts.zip {artifacts}/{run_id}/{build_environment}/artifacts.zip",
    "unzip -o -q artifacts.zip",
    "rm artifacts.zip",
    "mkdir -p .additional_ci_files",
    "curl -fsSL -o .additional_ci_files/td_results.json {artifacts}/{run_id}/td_results/td_results.json",
    "chown -R jenkins {workspace}",
    'sudo -E -H -u jenkins env "PATH=$PATH" python3 -m pip install "$(echo dist/*.whl)[opt-einsum]"',
]
# What a sandbox runs: the job's test step, as the jenkins user
JOB = r"""
set -euo pipefail
OUT=/out/$SHA1/$TEST_CONFIG-$SHARD_NUMBER
mkdir -p "$OUT"
exec > >(tee "$OUT/log.txt") 2>&1
# Modal sets these to the CPUs requested. test.sh picks a quarter of the CPUs it
# sees, at least 4: 4 on CI's runner, more here, where a sandbox sees the CPUs
# it may use
unset MKL_NUM_THREADS OPENBLAS_NUM_THREADS BLIS_NUM_THREADS
export OMP_NUM_THREADS=4
git config --global --add safe.directory "$GITHUB_WORKSPACE"
cd "$GITHUB_WORKSPACE"
mkdir -p "$HF_CACHE" "$HF_DATASETS_CACHE"
touch "$GITHUB_STEP_SUMMARY" "$GITHUB_ENV" "$GITHUB_OUTPUT"
chown -R jenkins "$RUNNER_TEMP"
code=0
sudo -E -H -u jenkins env "PATH=$PATH" .ci/pytorch/test.sh || code=$?
tar czf "$OUT/test-reports.tgz" test/test-reports || true
echo "$code" > "$OUT/exit_code"
exit "$code"
"""


def github(path: str) -> Any:
    with urllib.request.urlopen(
        f"https://api.github.com/repos/{REPO}/{path}"
    ) as response:
        return json.load(response)


def trunk_run(sha: str) -> int:
    """The trunk.yml run that built the commit."""
    runs = github(f"actions/workflows/trunk.yml/runs?head_sha={sha}")["workflow_runs"]
    if not runs:
        sys.exit(f"trunk.yml has no run for {sha}")
    return int(runs[0]["id"])


def image(sha: str, run_id: int) -> modal.Image:
    docker = next(
        e["sha"] for e in github(f"contents/.ci?ref={sha}") if e["name"] == "docker"
    )
    names = {
        "workspace": WORKSPACE,
        "repo": REPO,
        "sha": sha,
        "artifacts": ARTIFACTS,
        "run_id": run_id,
        "build_environment": BUILD_ENVIRONMENT,
    }
    setup = " && ".join(step.format(**names) for step in SETUP)
    return modal.Image.from_registry(f"{CI_IMAGE}-{docker}").run_commands(
        f"bash -c {shlex.quote(setup)}"
    )


def environment(sha: str, run_id: int, config: str, shard: int) -> dict[str, str]:
    """What _linux-test.yml gives the job's test step."""
    shards = MATRIX[config]
    return {
        "BUILD_ENVIRONMENT": BUILD_ENVIRONMENT,
        "TEST_CONFIG": config,
        "SHARD_NUMBER": str(shard),
        "NUM_TEST_SHARDS": str(shards),
        "TESTS_TO_EXCLUDE": TESTS_TO_EXCLUDE,
        "PYTEST_ADDOPTS": " ".join(f"--deselect {test}" for test in DESELECTED),
        "SHA1": sha,
        "BASE_SHA": sha,
        "BRANCH": "main",
        "GIT_DEFAULT_BRANCH": "main",
        "CI": "true",
        "GITHUB_ACTIONS": "true",
        "GITHUB_REPOSITORY": REPO,
        "GITHUB_WORKFLOW": "trunk",
        "GITHUB_JOB": "test",
        "GITHUB_RUN_ID": str(run_id),
        "GITHUB_RUN_NUMBER": "1",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_WORKSPACE": WORKSPACE,
        "GITHUB_STEP_SUMMARY": f"{RUNNER_TEMP}/summary",
        "GITHUB_ENV": f"{RUNNER_TEMP}/env",
        "GITHUB_OUTPUT": f"{RUNNER_TEMP}/output",
        "RUNNER_TEMP": RUNNER_TEMP,
        "JOB_ID": "0",
        "JOB_NAME": f"{BUILD_ENVIRONMENT} / test ({config}, {shard}, {shards}, modal-l4)",
        "USE_ARC": "1",
        "CONTINUE_THROUGH_ERROR": "1",
        "NO_TD": "False",
        "TD_DISTRIBUTED": "False",
        "PYTORCH_TEST_RERUN_DISABLED_TESTS": "0",
        "SCCACHE_BUCKET": "ossci-compiler-cache-circleci-v2",
        "SCCACHE_REGION": "us-east-1",
        "SCCACHE_S3_NO_CREDENTIALS": "true",
        "SKIP_SCCACHE_INITIALIZATION": "1",
        "HF_CACHE": f"{RUNNER_TEMP}/hf_cache",
        "HF_DATASETS_CACHE": f"{RUNNER_TEMP}/hf_datasets",
        "TRANSFORMERS_OFFLINE": "0",
        "HF_DATASETS_OFFLINE": "0",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.partition("\n")[0])
    parser.add_argument(
        "sha", help="a commit of pytorch/pytorch that trunk.yml has built"
    )
    parser.add_argument(
        "--only", nargs="+", metavar="CONFIG:SHARD", help="run these jobs only"
    )
    parser.add_argument(
        "--run-id", type=int, help="the trunk.yml run to take the build from"
    )
    options = parser.parse_args()
    jobs = [
        (config, shard)
        for config, shards in MATRIX.items()
        for shard in range(1, shards + 1)
    ]
    if options.only:
        jobs = [job for job in jobs if f"{job[0]}:{job[1]}" in options.only]
    if not jobs:
        parser.error("--only names no L4 job")
    sha = options.sha
    run_id = options.run_id or trunk_run(sha)
    app = modal.App.lookup(NAME, create_if_missing=True)
    volume = modal.Volume.from_name(NAME, create_if_missing=True)
    built = image(sha, run_id)
    print(f"trunk run {run_id}: building the image", flush=True)
    with modal.enable_output():  # once, before the jobs share it
        modal.Sandbox.create("true", app=app, image=built).wait()
    start = time.monotonic()

    def run(job: tuple[str, int]) -> tuple[int | None, float]:
        config, shard = job
        sandbox = modal.Sandbox.create(
            "bash",
            "-c",
            JOB,
            app=app,
            image=built,
            gpu=GPU,
            cpu=CPUS,
            memory=MEMORY_MIB,
            timeout=TIMEOUT_SECONDS,
            volumes={"/out": volume},
            env=environment(sha, run_id, config, shard),
        )
        print(
            f"{config} {shard}/{MATRIX[config]}: sandbox {sandbox.object_id}",
            flush=True,
        )
        sandbox.wait(raise_on_termination=False)
        minutes = (time.monotonic() - start) / 60
        print(
            f"{config} {shard}/{MATRIX[config]}: exit code {sandbox.returncode} after {minutes:.0f} min",
            flush=True,
        )
        return sandbox.returncode, minutes

    with ThreadPoolExecutor(len(jobs)) as pool:
        results = list(pool.map(run, jobs))
    failed = [
        f"{config}:{shard}"
        for (config, shard), (code, _) in zip(jobs, results)
        if code != 0
    ]
    took = max(minutes for _, minutes in results)
    print(
        f"{len(jobs) - len(failed)} of {len(jobs)} jobs passed in {took:.0f} min; logs: modal volume ls {NAME} {sha}"
    )
    if failed:
        print("failed: " + " ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
