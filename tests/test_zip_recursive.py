import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from zip_recursive import ZipExtractionLimits, extract_zip_recursive, write_extracted_files


def _make_zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
    return buf.getvalue()


class ZipRecursiveTests(unittest.TestCase):
    def test_extracts_nested_zip_recursively(self):
        current = _make_zip({"leaf.txt": b"ok"})
        for i in range(11):
            current = _make_zip({f"nested_{i}.zip": current})

        result = extract_zip_recursive(
            archive_name="root.zip",
            archive_bytes=current,
            limits=ZipExtractionLimits(max_depth=12, max_files=200, max_total_bytes=5 * 1024 * 1024),
        )

        self.assertEqual(result.extracted_files, 1)
        self.assertTrue(any(name.endswith("leaf.txt") for name, _ in result.files))
        self.assertEqual(len(result.warnings), 0)

    def test_blocks_path_traversal_entries(self):
        payload = _make_zip({"../evil.txt": b"x", "ok.txt": b"y"})

        result = extract_zip_recursive(archive_name="a.zip", archive_bytes=payload)

        self.assertEqual(result.extracted_files, 1)
        self.assertTrue(any(name.endswith("ok.txt") for name, _ in result.files))
        self.assertTrue(any("dangerous-path" in warn for warn in result.warnings))

    def test_write_files_preserves_existing_by_suffix(self):
        payload = _make_zip({"same.txt": b"one", "dir/same.txt": b"two"})
        result = extract_zip_recursive(archive_name="a.zip", archive_bytes=payload)

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            first = out / "root" / "same.txt"
            first.parent.mkdir(parents=True, exist_ok=True)
            first.write_bytes(b"existing")

            paths = write_extracted_files(out, result.files)
            self.assertGreaterEqual(len(paths), 2)
            self.assertTrue(any(p.name.startswith("same") for p in paths))


if __name__ == "__main__":
    unittest.main()
