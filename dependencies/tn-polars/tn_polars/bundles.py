"""Directory and ZIP storage that preserves edition file bytes."""

from contextlib import contextmanager
from dataclasses import dataclass
import io
from pathlib import Path
import zipfile

from .release import PolicyDenied


@contextmanager
def _open_archive(path):
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len({info.filename for info in infos}) != len(infos):
                raise PolicyDenied("edition archive contains duplicate member names")
            if any(info.compress_type != zipfile.ZIP_STORED for info in infos):
                raise PolicyDenied("edition archives require uncompressed members")
            yield archive
    except (zipfile.BadZipFile, KeyError) as error:
        raise PolicyDenied("edition archive is corrupt or incomplete") from error


@dataclass(frozen=True)
class PackedArtifactPath:
    archive_path: Path
    member: str

    def read_bytes(self):
        with _open_archive(self.archive_path) as archive:
            return archive.read(self.member)


def read_artifact_files(*paths):
    """Read a group of current artifact files with one open when packed."""
    if paths and all(isinstance(path, PackedArtifactPath) for path in paths):
        archive_path = paths[0].archive_path
        if all(path.archive_path == archive_path for path in paths):
            with _open_archive(archive_path) as archive:
                return tuple(archive.read(path.member) for path in paths)
    return tuple(path.read_bytes() for path in paths)


class _Bundle:
    def __init__(self, root, archive=None):
        self.root, self.archive = root, archive

    def read(self, name):
        return (self.root / name).read_bytes() if self.archive is None else self.archive.read(name)

    def is_file(self, name):
        if self.archive is None:
            return (self.root / name).is_file()
        info = self.archive.NameToInfo.get(name)
        return info is not None and not info.is_dir()

    def member(self, name):
        if self.archive is None:
            return self.root / name
        return PackedArtifactPath(self.root / "bundle.zip", name)


@contextmanager
def open_bundle(root):
    root = Path(root)
    if (root / "bundle.zip").exists():
        if (root / "bundle.json").exists():
            raise PolicyDenied("edition directory contains conflicting bundle formats")
        with _open_archive(root / "bundle.zip") as archive:
            yield _Bundle(root, archive)
    else:
        yield _Bundle(root)


def pack_files(files):
    """Deterministic archive bytes make immutable retries byte-exact."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, raw in sorted(files.items()):
            # ZipInfo's fixed default timestamp avoids a different retry representation.
            archive.writestr(zipfile.ZipInfo(name), raw)
    return buffer.getvalue()
