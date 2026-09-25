import os.path
import numpy as np
from astropy.time import Time

from .config import PACKAGE_DATA_PATH
from .dc18 import get_ephemerides as get_dc18_ephemerides

# ============================================================================
# Public API
# ============================================================================


def get_kwargs(filename):
    """
    Parse the filename to create a dict of kwargs for MulensModel.MulensData.

    Parameters
    ----------
    filename : str
        Format: nYYYYMMDD.BAND.TELESCOPE.whateveryouwant

    Returns
    -------
    dict
        Kwargs for MulensData constructor
    """
    telescope, band = get_telescope_band_from_filename(filename)

    # Use filename basename as unique label
    label = os.path.basename(filename)

    if telescope in OBSERVATORIES:
        obs = OBSERVATORIES[telescope]
        kwargs = obs.get_kwargs()
        kwargs["bandpass"] = band
        kwargs["plot_properties"] = obs.get_plot_properties(band)
        kwargs["telescope"] = telescope
        # Override label with filename
        kwargs["plot_properties"]["label"] = label
    else:
        # Default for unknown telescopes
        kwargs = {
            "phot_fmt": "flux",
            "bandpass": band,
            "plot_properties": {"label": label, "marker": "o"},
        }

    return kwargs


def get_telescope_band_from_filename(filename):
    """
    Get the telescope name and band from the filename.

    Parameters
    ----------
    filename : str
        Format: nYYYYMMDD.BAND.TELESCOPE.whateveryouwant

    Returns
    -------
    tuple
        (telescope, band)
    """
    if filename[-1] == ".":
        filename += " "

    basename = os.path.basename(filename).split(".")
    if len(basename) < 4:
        raise ValueError(
            f"Filename ({filename}) must have the format "
            + "nYYYYMMDD.BAND.TELESCOPE.whateveryouwant"
        )

    band = basename[1]
    telescope = basename[2]
    return telescope, band

def set_filename_from_MulensData(mulensdata, suffix="MMEXOFAST"):
    """
    Construct a filename stored in the MulensData.plot_properties object.
    Parameters
    ----------
    mulensdata : MulensData
        MulensData object
    suffix : str, optional
        Suffix to append to the filename (default: "")
    """
    date = Time(
        mulensdata.time[np.argmax(mulensdata.flux)], format="jd"
    ).strftime("%Y%m%d")

    telescope = getattr(mulensdata, "telescope", None)
    band = getattr(mulensdata, "bandpass", None)

    if telescope is None or band is None:
        label = mulensdata.plot_properties.get("label")
        if label is not None:
            parsed_telescope, parsed_band = get_telescope_band_from_filename(label)
            telescope = telescope if telescope is not None else parsed_telescope
            band = band if band is not None else parsed_band

    if telescope is None or band is None:
        raise ValueError(
            "Cannot determine telescope and band for MulensData; "
            "set telescope/bandpass or use a filename label in the format "
            "nYYYYMMDD.BAND.TELESCOPE.whateveryouwant."
        )

    mulensdata._telescope = telescope
    mulensdata.bandpass = band
    mulensdata.plot_properties["label"] = (
        f"n{date}.{band}.{telescope}.{suffix}.dat".rstrip(".")
    )


# ============================================================================
# Observatory Class and Registry
# ============================================================================


class Observatory:
    """
    Container for observatory-specific MulensData configuration.

    Parameters
    ----------
    name : str
        Observatory name
    phot_fmt : str, optional
        Photometry format ('flux' or 'mag')
    usecols : list, optional
        Columns to read from data file
    ephemerides_file : str or None, optional
        Path to ephemerides file for space-based observatories
    ephemerides_loader : callable or None, optional
        Zero-argument callable returning the path to the ephemerides file,
        for observatories whose ephemerides has to be located or fetched at
        use time rather than named up front. Called by :meth:`get_kwargs`
        only when ``ephemerides_file`` is None, so that constructing or
        registering an observatory never triggers the work. Ignored if
        ``ephemerides_file`` is given.
    bands : dict, optional
        Dict mapping band names to plot properties dicts
    """

    def __init__(
        self,
        name,
        phot_fmt="flux",
        usecols=None,
        ephemerides_file=None,
        ephemerides_loader=None,
        bands=None,
    ):
        self.name = name
        self.phot_fmt = phot_fmt
        self.usecols = usecols if usecols is not None else [0, 1, 2]
        self.ephemerides_file = ephemerides_file
        self.ephemerides_loader = ephemerides_loader
        self.bands = bands if bands is not None else {}

    def get_kwargs(self):
        """
        Get kwargs dict for MulensData creation.

        Raises
        ------
        FileNotFoundError
            If this observatory needs an ephemerides file that cannot be
            found, rather than passing a nonexistent path on to MulensModel.
        """
        kwargs = {"phot_fmt": self.phot_fmt}
        if self.usecols is not None:
            kwargs["usecols"] = self.usecols

        ephemerides_file = self.ephemerides_file
        if ephemerides_file is None and self.ephemerides_loader is not None:
            ephemerides_file = self.ephemerides_loader()

        if ephemerides_file is not None:
            if not os.path.isfile(ephemerides_file):
                raise FileNotFoundError(
                    "Observatory {0!r} needs the ephemerides file {1!r}, "
                    "which does not exist. Pass ephemerides_file to "
                    "MulensData yourself to point at your own copy.".format(
                        self.name, ephemerides_file
                    )
                )

            kwargs["ephemerides_file"] = ephemerides_file

        return kwargs

    def get_plot_properties(self, band):
        """Get plot properties for a specific band."""
        default = {"label": f"{self.name}-{band}", "marker": "o"}
        if band in self.bands:
            return {**default, **self.bands[band]}
        return default


# Observatory registry (public)
OBSERVATORIES = {}


def register_observatory(observatory):
    """Register an observatory instance."""
    OBSERVATORIES[observatory.name] = observatory


# Create reverse mapping from ephemerides_file to observatory name
EPHEMERIDES_TO_OBSERVATORY = {
    obs.ephemerides_file: name
    for name, obs in OBSERVATORIES.items()
    if obs.ephemerides_file is not None
}

# ============================================================================
# Utility Functions
# ============================================================================


def list_observatories():
    """
    List all registered observatories.

    Returns
    -------
    list
        List of observatory names
    """
    return list(OBSERVATORIES.keys())


def get_observatory(name):
    """
    Get an observatory by name.

    Parameters
    ----------
    name : str
        Observatory name

    Returns
    -------
    Observatory or None
        Observatory instance if registered, None otherwise
    """
    return OBSERVATORIES.get(name)


def validate_filename(filename):
    """
    Check if filename follows the expected format.

    Parameters
    ----------
    filename : str
        Filename to validate

    Returns
    -------
    bool
        True if valid, False otherwise
    """
    try:
        get_telescope_band_from_filename(filename)
        return True
    except ValueError:
        return False


def load_observatories_from_config(config_file):
    """
    Load and register observatories from a configuration file.

    Parameters
    ----------
    config_file : str
        Path to YAML or JSON config file

    Notes
    -----
    Expected format (YAML):
        observatories:
          - name: MyTelescope
            phot_fmt: mag
            bands:
              V:
                color: blue
                marker: s

    Expected format (JSON):
        {
          "observatories": [
            {
              "name": "MyTelescope",
              "phot_fmt": "mag",
              "bands": {
                "V": {"color": "blue", "marker": "s"}
              }
            }
          ]
        }
    """
    import json

    # Try JSON first
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
    except json.JSONDecodeError:
        # Try YAML
        try:
            import yaml

            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
        except ImportError:
            raise ImportError(
                "YAML support requires PyYAML: pip install pyyaml"
            )

    # Register observatories from config
    for obs_config in config.get("observatories", []):
        obs = Observatory(
            name=obs_config["name"],
            phot_fmt=obs_config.get("phot_fmt", "flux"),
            usecols=obs_config.get("usecols"),
            ephemerides_file=obs_config.get("ephemerides_file"),
            bands=obs_config.get("bands", {}),
        )
        register_observatory(obs)


# ============================================================================
# Built-in Observatories
# ============================================================================
# Fake


register_observatory(
    Observatory(
        name="WFIRST18",
        phot_fmt="flux",
        usecols=[0, 1, 2],
        ephemerides_loader=get_dc18_ephemerides,
        bands={
            "W149": {"color": "darkorange", "marker": "o"},
            "Z087": {"color": "darkcyan", "marker": "s", "zorder": 5},
        },
    )
)

register_observatory(
    Observatory(
        name="DC18",
        phot_fmt="mag",
        usecols=[0, 1, 2],
        ephemerides_loader=get_dc18_ephemerides,
        bands={
            "W149": {"color": "darkorange", "marker": "o"},
            "Z087": {"color": "darkcyan", "marker": "s", "zorder": 5},
        },
    )
)

# Space-based
register_observatory(
    Observatory(
        name="Spitzer",
        phot_fmt="flux",
        usecols=[0, 1, 2],
        ephemerides_file=os.path.join(
            PACKAGE_DATA_PATH, "spitzer_ephemerides_2014_to_2019.txt"
        ),
        bands={"L": {"color": "red", "marker": "o"}},
    )
)

# Ground-based
register_observatory(
    Observatory(
        name="OGLE",
        phot_fmt="flux",
        usecols=[0, 1, 2],
        bands={
            "I": {"color": "black", "marker": "o"},
            "V": {"color": "black", "marker": "v", "facecolor": "none"},
        },
    )
)
