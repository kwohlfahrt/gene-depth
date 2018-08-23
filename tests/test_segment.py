from gene_depth.main import main
from tifffile import TiffWriter, TiffFile
import numpy as np

from click.testing import CliRunner
import pytest


@pytest.fixture
def runner():
    return CliRunner()


def test_commandline(tmpdir, runner):
    input = tmpdir / 'in.tif'
    output = tmpdir / 'out.tif'

    data = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 1, 1, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype='uint32')
    TiffWriter(str(input)).save(data * 5)

    result = runner.invoke(main, ["segment", "run"] + list(map(str, [input, output, '--threshold', 2])))
    assert result.exit_code == 0

    r = TiffFile(str(output)).asarray()
    assert r[2,8] == 0
    assert r[1,2] > 0
    assert r[8,8] > 0
    assert r[1,2] != r[8,8]
