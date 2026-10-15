"""Check import, CLAP stats, and tensor copies in a normal or fast build."""

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-root", type=Path,
        default=Path(__file__).resolve().parents[1] / "python",
        help="directory containing the torch_paras package to check",
    )
    parser.add_argument("--expect-clap", choices=("on", "off"), required=True)
    parser.add_argument("--device", default="paras:0")
    args = parser.parse_args()
    sys.path.insert(0, str(args.package_root.resolve()))

    import torch
    import torch_paras

    extension = torch_paras._C
    assert Path(extension.__file__).resolve().parent == (
        args.package_root / "torch_paras"
    ).resolve(), f"Imported unexpected extension: {extension.__file__}"
    assert extension.clap_dnn_enabled() is (args.expect_clap == "on")

    def check_stats():
        extension.clap_dnn_reset_stats()
        assert extension.clap_dnn_call_count() == 0
        assert extension.clap_dnn_fallback_count() == 0
        assert extension.clap_dnn_last_backend() == ""
        for backend in ("CPU", "oneDNN", "CUDA", "cuDNN", "ROCM", "AMD", "MIOpen"):
            assert extension.clap_dnn_backend_call_count(backend) == 0

    check_stats()
    source = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    copied = source.to(args.device)
    assert copied.device == torch.device(args.device)
    torch.testing.assert_close(copied.cpu(), source, atol=0, rtol=0)
    check_stats()
    print(f"PASS: {extension.__file__}, CLAP={args.expect_clap}, {args.device} copy")


if __name__ == "__main__":
    main()
