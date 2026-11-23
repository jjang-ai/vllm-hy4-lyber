"""CPU (TRITON_INTERPRET=1) check of the DFly query-slot kernel under DCP.

Loads `_dfly_query_slots_kernel` from the given dfly_inputs.py without
importing vLLM and compares it against the target's slot layout
(`_compute_slot_mappings_kernel` math in v1/worker/gpu/block_table.py) and the
read-side conversion in sparse_utils.py (owner=(tok//I)%D,
local=(tok//(D*I))*I + tok%I). Usage: python test.py <dfly_inputs.py> [old]
"""

import ast
import os
import sys

os.environ["TRITON_INTERPRET"] = "1"
import numpy as np  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

PAD = -1


def load_kernel(path):
    tree = ast.parse(open(path).read())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_dfly_query_slots_kernel"
    )
    mod = ast.Module(body=[fn], type_ignores=[])
    src = ast.unparse(mod)
    ns = {"triton": triton, "tl": tl}
    # Interpreter needs real source lines: write to a file and exec from it.
    tmp = "/tmp/_dfly_kernel_under_test.py"
    with open(tmp, "w") as f:
        f.write("import triton\nimport triton.language as tl\n" + src + "\n")
    code = compile(open(tmp).read(), tmp, "exec")
    exec(code, ns)
    return ns["_dfly_query_slots_kernel"], len(fn.args.args)


def reference(pos, block_row, n_alloc, B, D, rank, I, sharded=True):
    if D == 1 or not sharded:
        local, owner = pos, True
    else:
        owner = (pos // I) % D == rank
        local = (pos // (D * I)) * I + pos % I
    col = local // B
    if not owner or col >= n_alloc or block_row[col] == 0:
        return PAD
    return int(block_row[col]) * B + local % B


def run(path, old=False):
    kernel, nargs = load_kernel(path)
    rng = np.random.default_rng(0)
    B, QW, max_len = 64, 6, 1 << 20
    total_checked = 0
    for D, I in ((1, 1), (8, 1), (8, 4), (8, 64), (4, 16)):
        reqs = 24
        cap = -(-max_len // (B * D))  # block-table width under DCP
        anchors = rng.integers(1, max_len // 2, reqs)
        anchors[:4] = [1, 63, 511, 4093]
        active = rng.random(reqs) > 0.2
        idx_mapping = np.where(active, rng.permutation(64)[:reqs], -1)
        n_alloc_state = np.zeros(64, np.int32)
        bt = np.zeros((reqs, cap), np.int32)
        for r in range(reqs):
            need = (anchors[r] + QW - 1) // (B * D) + 1
            n = min(cap, need + rng.integers(0, 2))
            if idx_mapping[r] >= 0:
                n_alloc_state[idx_mapping[r]] = n
            bt[r, :n] = rng.integers(1, 1 << 20, n)
            # Stale ids of an earlier, longer occupant of this batch row.
            bt[r, n : n + 4000] = rng.integers(1, 1 << 20, min(4000, cap - n))
        positions = np.zeros(reqs * QW, np.int64)
        for r in range(reqs):
            positions[r * QW : (r + 1) * QW] = (
                anchors[r] + np.arange(QW) if active[r] else np.arange(QW)
            )
        owners = {}
        for rank in range(D):
            out = torch.full((reqs * QW,), -7, dtype=torch.int64)
            common = dict(
                QUERY_WIDTH=QW,
                QUERY_BLOCK=triton.next_power_of_2(QW),
                PAD_SLOT_ID=PAD,
            )
            if old:
                kernel[(reqs,)](
                    out,
                    torch.from_numpy(positions),
                    torch.from_numpy(active),
                    torch.from_numpy(bt),
                    bt.shape[1],
                    bt.shape[1],
                    B,
                    torch.tensor(max_len, dtype=torch.int32),
                    **common,
                )
            else:
                kernel[(reqs,)](
                    out,
                    torch.from_numpy(positions),
                    torch.from_numpy(active),
                    torch.from_numpy(idx_mapping.astype(np.int64)),
                    torch.from_numpy(bt),
                    bt.shape[1],
                    bt.shape[1],
                    torch.from_numpy(n_alloc_state),
                    B,
                    torch.tensor(max_len, dtype=torch.int32),
                    torch.tensor([True]),
                    rank,
                    CP_SIZE=D,
                    CP_INTERLEAVE=I,
                    **common,
                )
            out = out.numpy()
            bad = 0
            foreign = 0
            for r in range(reqs):
                n_alloc = n_alloc_state[idx_mapping[r]] if idx_mapping[r] >= 0 else 0
                own = set(bt[r, :n_alloc].tolist())
                for j in range(QW):
                    v = out[r * QW + j]
                    if v != PAD and v // B not in own:
                        foreign += 1
            for r in range(reqs):
                n_alloc = n_alloc_state[idx_mapping[r]] if idx_mapping[r] >= 0 else 0
                for j in range(QW):
                    t = r * QW + j
                    exp = (
                        reference(positions[t], bt[r], n_alloc, B, D, rank, I)
                        if active[r]
                        else PAD
                    )
                    if exp != PAD:
                        owners.setdefault(t, []).append(rank)
                    if out[t] != exp:
                        bad += 1
                    total_checked += 1
            print(f"D={D} I={I} rank={rank}: mismatches vs target layout = {bad}, writes into blocks the request does not own = {foreign}")
            if not old:
                assert bad == 0
        if not old:
            # Every active token is written exactly once across the DCP group.
            for r in range(reqs):
                if active[r]:
                    for j in range(QW):
                        assert owners.get(r * QW + j) == [
                            (positions[r * QW + j] // I) % D
                        ], (r, j)
    print("checked", total_checked, "token slots;", "OLD kernel" if old else "OK")


if __name__ == "__main__":
    run(sys.argv[1], old=len(sys.argv) > 2 and sys.argv[2] == "old")
