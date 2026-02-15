import argparse
from pathlib import Path

import lance
import pyarrow.dataset as ds
from tqdm import tqdm


def _selected_parquet_files(parquet_dir: Path, include_prefixes: tuple[str, ...]) -> list[Path]:
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

    selected = [p for p in parquet_files if p.stem.startswith(include_prefixes)]
    if not selected:
        raise FileNotFoundError(
            f"No parquet files matched prefixes {include_prefixes} in {parquet_dir}"
        )
    return selected


def convert_files(
    parquet_dir: Path,
    output_root: Path,
    include_prefixes: tuple[str, ...],
    overwrite: bool,
    dry_run: bool,
) -> None:
    selected = _selected_parquet_files(parquet_dir, include_prefixes)
    output_root.mkdir(parents=True, exist_ok=True)
    mode = "overwrite" if overwrite else "create"

    print(f"Input parquet dir: {parquet_dir}")
    print(f"Output lance dir:  {output_root}")
    print(f"Prefixes:          {include_prefixes}")
    print(f"Mode:              {mode}")
    print(f"Files to convert:  {len(selected)}")

    for parquet_path in tqdm(selected, desc="Converting", unit="file"):
        out_path = output_root / f"{parquet_path.stem}.lance"
        tqdm.write(f"{parquet_path.name} -> {out_path.name}")
        if dry_run:
            continue
        dataset = ds.dataset(str(parquet_path), format="parquet")
        lance.write_dataset(dataset, str(out_path), mode=mode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MKB2 parquet shards to per-file Lance datasets."
    )
    parser.add_argument(
        "--parquet-dir",
        default="/proj/bedrock/datasets/MKB2/parquet/processed",
        help="Directory containing MKB2 parquet shards.",
    )
    parser.add_argument(
        "--output-root",
        default="/proj/bedrock/datasets/MKB2/lance",
        help="Output directory for .lance datasets.",
    )
    parser.add_argument(
        "--include-prefixes",
        nargs="+",
        default=["mkb_train_", "mkb_val"],
        help="Parquet filename prefixes to include. small_* is ignored by default.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing lance outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected files without writing output.",
    )
    args = parser.parse_args()

    parquet_dir = Path(args.parquet_dir)
    output_root = Path(args.output_root)
    prefixes = tuple(args.include_prefixes)

    convert_files(
        parquet_dir=parquet_dir,
        output_root=output_root,
        include_prefixes=prefixes,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
