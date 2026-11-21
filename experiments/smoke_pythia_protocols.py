"""Exercise both two-process wire protocols without downloading model weights."""
import numpy as np

from pythia_qk_2pc import evaluate_scores, workers
from pythia_secure_head import project, projection_workers


def test_head():
    matrix = (np.arange(7 * 5, dtype=np.int64).reshape(7, 5) - 13) * (1 << 23)
    hidden = np.array([-34, 0, 56, 20, -11], dtype=np.int64) * (1 << 23)
    ids = np.array([5, 1, 4, 0], dtype=np.int64)
    with projection_workers(matrix) as channels:
        for restrict, public in ((False, True), (True, True), (True, False)):
            output, upload, download, disclosed, _ = project(
                channels, hidden, ids, len(matrix), restrict, public)
            selected = len(ids) if restrict and public else 0
            np.testing.assert_array_equal(output, matrix[ids] @ hidden)
            assert disclosed == selected
            assert upload == 2 * (4 + 4 * selected + hidden.nbytes)
            assert download == 16 * (selected or len(matrix))


def test_qk():
    qk = (np.arange(2 * 5 * 4, dtype=np.int64).reshape(2, 5, 4) - 21) * (1 << 21)
    expected = (qk[0] @ qk[1].T)[np.tril_indices(5)]
    with workers("tcp") as channels:
        for public_prefix in (0, 3, 5):
            output, counts = evaluate_scores(channels, qk.view(np.uint64), public_prefix)
            np.testing.assert_array_equal(output, expected)
            suffix = 5 - public_prefix
            products = 4 * suffix * (suffix + 1) // 2
            assert counts["beaver_multiplications"] == products
            assert counts["peer_payload_bytes"] == 32 * products


if __name__ == "__main__":
    test_head()
    test_qk()
    print("PASS: projection authorization fallback and causal QK over two TCP workers")
