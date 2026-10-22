"""Simple HTTP client to interact with the server."""

from typing import Union, List, Type, Optional

import tempfile
import os
import urllib.parse

import requests
from pathlib import Path

import sisl
from ._serialization import prediction_suffix


class ServerClient:
    """Client to interact easily with the e3nn server.

    Parameters
    ----------
    url : str or None, optional
        Root url where the server is running.

        If it is set to None, the environment variable E3MAT_SERVER_URL
        will be used if present.

        If it is set to None and the environment variable is not present,
        the url will be constructed from the values of ``host`` and ``port``.
    host : str or None, optional
        Host where the server is running.

        If it is set to None, the environment variable E3MAT_SERVER_HOST
        will be used if present, otherwise the default value of ``host`` will
        be used.
    port : int, optional
        Port where the server is listening.

        If it is set to None, the environment variable E3MAT_SERVER_PORT
        will be used if present, otherwise the default value of ``port`` will
        be used.
    """

    host: Union[str, None]
    port: Union[int, None]
    root_url: str

    def __init__(
        self,
        url: Optional[str] = None,
        host: Optional[str] = "localhost",
        port: Optional[int] = 56000,
    ):
        url = url or os.environ.get("E3MAT_SERVER_URL", None)

        if url is None:
            self.host = host or os.environ.get("E3MAT_SERVER_HOST", "localhost")
            self.port = port or int(os.environ.get("E3MAT_SERVER_PORT", 56000))

            self.root_url = f"http://{self.host}:{self.port}"
        else:
            parsed_url = urllib.parse.urlparse(url)

            self.host = parsed_url.hostname
            self.port = parsed_url.port
            self.root_url = url

        self.api_url = f"{self.root_url}/api"

    def avail_models(self) -> List[str]:
        """Returns the models that are available in the server."""
        response = requests.get(f"{self.api_url}/avail_models")
        response.raise_for_status()
        return response.json()

    def predict(
        self,
        geometry: Union[str, Path, sisl.Geometry],
        output: Union[str, Path, Type[sisl.SparseOrbital]],
        model: str,
        local: bool = False,
    ) -> Union[Path, sisl.SparseOrbital]:
        """Predicts the matrix for a given geometry.

        Parameters
        ----------
        geometry : Union[str, sisl.Geometry]
            Either the path to the geometry file or the geometry itself.
        output : Union[str, Type[sisl.SparseOrbital]]
            Either the path to the file where the prediction should be saved
            or the type of the object to be returned.
        model : str
            Name of the model to use for the prediction. This model must be available
            in the server. You can check the available models with the `avail_models` method.
        local : bool
            Whether the paths (if given) are in the local filesystem.
        """
        as_matrix = isinstance(output, type) and issubclass(output, sisl.SparseOrbital)
        with tempfile.TemporaryDirectory(prefix="g2m-client-") as temporary:
            temporary = Path(temporary)
            if isinstance(geometry, sisl.Geometry):
                geometry_path = temporary / "geometry.xyz"
                geometry.write(geometry_path)
            else:
                geometry_path = Path(geometry).absolute()

            if as_matrix:
                kind = {sisl.DensityMatrix: "density_matrix", sisl.Hamiltonian: "hamiltonian",
                        sisl.EnergyDensityMatrix: "energy_density_matrix"}[output]
                output_path = temporary / ("prediction" + prediction_suffix(kind))
            else:
                output_path = Path(output).absolute()

            if local:
                response = requests.get(
                    f"{self.api_url}/models/{model}/local_write_predict",
                    params={"geometry_path": str(geometry_path), "output_path": str(output_path)},
                )
                response.raise_for_status()
            else:
                with open(geometry_path, "rb") as geometry_file:
                    response = requests.post(
                        f"{self.api_url}/models/{model}/predict",
                        params={"output_suffix": output_path.suffix},
                        files={"geometry_file": geometry_file},
                    )
                response.raise_for_status()
                output_path.write_bytes(response.content)

            # DM binary files omit coordinates/basis: use the supplied geometry (read from
            # its file when a path was given), or sisl would invent one.
            if not isinstance(geometry, sisl.Geometry):
                geometry = sisl.get_sile(str(geometry_path)).read_geometry()
            read_kwargs = {"geometry": geometry}
            return output.read(output_path, **read_kwargs) if as_matrix else output_path
