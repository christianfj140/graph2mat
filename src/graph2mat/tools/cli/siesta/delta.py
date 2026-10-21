from pathlib import Path
from typing import List, Optional

import typer

from graph2mat.tools.siesta.delta import add_h0, compute_h0, write_delta_sample
from graph2mat.tools.siesta.runner import system_label

app = typer.Typer(
    help="Delta-learning on the non-self-consistent SIESTA H0 (H - H0 labels, H0 added back).",
    rich_markup_mode="markdown",
)


@app.command("h0")
def h0_command(
    reference_dirs: List[Path] = typer.Argument(..., help="Converged SIESTA runs (RUN.fdf, pseudopotentials)."),
    out_root: Path = typer.Option(..., help="H0 of each run goes to OUT_ROOT/<run directory name>."),
    siesta: Optional[str] = typer.Option(None, help="SIESTA command (default: $SIESTA or 'siesta' on the PATH)."),
):
    """Compute H0 (one SCF iteration from atomic densities) of each run."""
    for reference_dir in reference_dirs:
        typer.echo(compute_h0(reference_dir, out_root / reference_dir.name, siesta=siesta))


@app.command("label")
def label_command(
    reference_dir: Path = typer.Argument(..., help="Converged SIESTA run whose <SystemLabel>.TSHS is the label."),
    h0: Path = typer.Argument(..., help="Its H0 (TSHS)."),
    out_dir: Path = typer.Argument(..., help="Directory of the delta sample (RUN.fdf + H - H0)."),
):
    """Write a copy of a sample whose label is H - H0."""
    matrix = reference_dir / f"{system_label(reference_dir / 'RUN.fdf')}.TSHS"
    typer.echo(write_delta_sample(reference_dir, matrix, h0, out_dir))


@app.command("add-h0")
def add_h0_command(
    predicted: Path = typer.Argument(..., help="Prediction of H - H0 (HSX/TSHS)."),
    h0: Path = typer.Argument(..., help="H0 of the same structure (TSHS)."),
    out: Optional[Path] = typer.Option(None, help="Output file (default: overwrite PREDICTED)."),
):
    """Turn a prediction of H - H0 into a prediction of H."""
    typer.echo(add_h0(predicted, h0, out))


if __name__ == "__main__":
    app()
