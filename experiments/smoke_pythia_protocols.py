"""Exercise both two-process wire protocols without downloading model weights."""
from unittest.mock import Mock

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
        for public_prefix in (0, np.int64(3), 5):
            output, counts = evaluate_scores(channels, qk.view(np.uint64), public_prefix)
            np.testing.assert_array_equal(output, expected)
            suffix = 5 - public_prefix
            products = 4 * suffix * (suffix + 1) // 2
            assert counts["beaver_multiplications"] == products
            assert counts["peer_payload_bytes"] == 32 * products


def test_qk_rejects_invalid_metadata_without_sending():
    qk = np.zeros((2, 5, 4), dtype=np.uint64)
    channels = [Mock(spec=["sendall"]), Mock(spec=["sendall"])]
    cases = (
        (qk[0], 0, "array of shape"),
        (qk[:1], 0, "array of shape"),
        (qk.astype(np.float64), 0, "uint64 array"),
        (qk[:, :0, :], 0, "length and width"),
        (qk[:, :, :0], 0, "length and width"),
        (qk, -1, "public prefix must be in"),
        (qk, 6, "public prefix must be in"),
        (qk, 2.0, "public prefix must be an integer"),
        (qk, True, "public prefix must be an integer"),
    )
    for values, prefix, message in cases:
        with np.testing.assert_raises_regex(ValueError, message):
            evaluate_scores(channels, values, prefix)
    for channel in channels:
        channel.sendall.assert_not_called()


if __name__ == "__main__":
    test_qk_rejects_invalid_metadata_without_sending()
    test_head()
    test_qk()
    print("PASS: QK metadata validation, projection authorization fallback, and causal QK over TCP")
