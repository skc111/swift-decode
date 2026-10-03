import os
import unittest
from unittest.mock import Mock, patch

from tokenrush.backend import BACKENDS, select_backend


class BackendTests(unittest.TestCase):
    def test_explicit_backend_never_calls_auto_probe(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), patch.dict(os.environ, TOKENRUSH_BACKEND=backend):
                probe = Mock(side_effect=AssertionError("must not compile Marlin"))
                self.assertEqual(select_backend(probe), backend)
                probe.assert_not_called()

    def test_auto_preserves_upstream_selection(self):
        for available in (True, False):
            with self.subTest(available=available), patch.dict(os.environ, TOKENRUSH_BACKEND="auto"):
                self.assertEqual(select_backend(lambda: available), "marlin" if available else "triton")

    def test_unset_is_auto(self):
        with patch.dict(os.environ):
            os.environ.pop("TOKENRUSH_BACKEND", None)
            self.assertEqual(select_backend(lambda: True), "marlin")

    def test_invalid_backend_does_not_fall_back(self):
        with patch.dict(os.environ, TOKENRUSH_BACKEND="typo"), self.assertRaises(ValueError):
            select_backend(lambda: True)


if __name__ == "__main__":
    unittest.main()
