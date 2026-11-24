"""Build the single readable generator script required by the release contract."""
from pathlib import Path
import ast
import json

ROOT=Path(__file__).resolve().parent
WORK=Path("C:/Users/chrom/AppData/Local/Temp/binary-cat-mechanics")


def clean_module(path):
    text=path.read_text(encoding="utf-8")
    lines=text.splitlines(keepends=True)
    omitted=set()
    for node in ast.walk(ast.parse(text)):
        remove=False
        if isinstance(node,ast.ImportFrom) and node.module in ("__future__","mechanics","evaluate_mechanics"):
            remove=True
        if isinstance(node,ast.Import) and any(x.name in ("candidate_generation","mechanics_selection") for x in node.names):
            remove=True
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name=="main":
            remove=True
        if isinstance(node,ast.If) and isinstance(node.test,ast.Compare) and "__name__" in ast.unparse(node.test):
            remove=True
        if remove:
            omitted.update(range(node.lineno-1,node.end_lineno))
    return "".join(line for i,line in enumerate(lines) if i not in omitted)


def build(destination):
    header='''"""Reproduce binary cat specimens with FEM labels and balanced joint-property selection.

This standalone file contains generation, Q4 FEM, selection and validation.
Reference AI-review exclusions apply only to the exact candidate-pixel digest.
Black=hard material10, white=soft material1; both Poisson ratios=.3.
"""
from __future__ import annotations
import os
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"
import sys
gen = sys.modules[__name__]
selection = sys.modules[__name__]
'''
    parts=[header]
    for name in ("candidate_generation.py","mechanics.py","mechanics_selection.py","evaluate_mechanics.py","release_workflow.py"):
        parts += [f"\n\n# ----- {name} -----\n",clean_module(ROOT/name)]
    text="".join(parts)
    text=text.replace('default=Path(__file__).resolve().parent/"cat_binary_dataset"','default=Path(__file__).resolve().parent')
    reviews={}
    for mode in ("pilot","full"):
        path=WORK/f"{mode}_visual_review.json"
        if path.exists():reviews[mode]=json.loads(path.read_text(encoding="utf-8"))
    baseline=json.loads((WORK/"baseline_summary.json").read_text()) if (WORK/"baseline_summary.json").exists() else {}
    solver_path=Path("C:/Users/chrom/AppData/Local/Temp/cat-mechanics-validation.json")
    solver=json.loads(solver_path.read_text()) if solver_path.exists() else {}
    text+=f"\n\nREFERENCE_REVIEW = {reviews!r}\nREFERENCE_BASELINE = {baseline!r}\nREFERENCE_SOLVER_CHECKS = {solver!r}\n"
    text+='\nif __name__ == "__main__":\n    workflow_main()\n'
    compile(text,str(destination),"exec")
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(text,encoding="utf-8")
    print(f"Built {destination} ({len(text)} characters)")


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination",type=Path)
    build(parser.parse_args().destination)
