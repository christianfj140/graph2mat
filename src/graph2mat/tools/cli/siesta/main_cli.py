import typer

from .delta import app as delta_app
from .md import app as md_app

app = typer.Typer(
    help="Set of utilities to interface the machine learning models with SIESTA."
)

app.add_typer(md_app, name="md")
app.add_typer(delta_app, name="delta")
