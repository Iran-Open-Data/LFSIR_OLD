import functools
from pathlib import Path
from typing import Callable, Iterable, Literal, Any, get_args

from pydantic import BaseModel
import yaml


PACKAGE_DIRECTORY = Path(__file__).parents[1]
ROOT_DIRECTORT = Path().absolute()

_Years = int | Iterable[int] | str | Literal["all", "last"]

_OriginalTable = Literal["data"]
_StandardTable = Literal[""]
_Table = _OriginalTable | _StandardTable

original_tables: tuple[_OriginalTable] = get_args(_OriginalTable)  # type: ignore

_Attribute = Literal["Year", "Season", "Urban_Rural", "Province"]

_Province = Literal[
    "Markazi",
    "Gilan",
    "Mazandaran",
    "East_Azerbaijan",
    "West_Azerbaijan",
    "Kermanshah",
    "Khuzestan",
    "Fars",
    "Kerman",
    "Razavi_Khorasan",
    "Isfahan",
    "Sistan_and_Baluchestan",
    "Kurdistan",
    "Hamadan",
    "Chaharmahal_and_Bakhtiari",
    "Lorestan",
    "Ilam",
    "Kohgiluyeh_and_Boyer-Ahmad",
    "Bushehr",
    "Zanjan",
    "Semnan",
    "Yazd",
    "Hormozgan",
    "Tehran",
    "Ardabil",
    "Qom",
    "Qazvin",
    "Golestan",
    "North_Khorasan",
    "South_Khorasan",
    "Alborz",
]


def open_yaml(
    path: Path | str,
    location: Literal["package", "root"] = "package",
    interpreter: Callable[[str], str] | None = None,
):
    """Open and parse a YAML file from package or root directory.

    Handles locating the YAML file based on provided path and 
    directory location. Runs an optional string interpreter 
    function before loading the YAML.

    Parameters
    ----------
    path : Path or str
        Path to YAML file.
    location : str, default "package"
        "package" or "root" directory location.
    interpreter : callable, optional
        Function to preprocess YAML string before loading.

    Returns
    -------
    dict
        Parsed YAML contents as a dictionary.
    """
    path = Path(path) if isinstance(path, str) else path
    if path.is_absolute():
        pass
    elif location == "root":
        path = ROOT_DIRECTORT.joinpath(path)
    else:
        path = PACKAGE_DIRECTORY.joinpath(path)

    with open(path, mode="r", encoding="utf8") as yaml_file:
        yaml_text = yaml_file.read()
    if interpreter is not None:
        yaml_text = interpreter(yaml_text)
    yaml_content = yaml.safe_load(yaml_text)
    return yaml_content


def flatten_dict(dictionary: dict) -> dict[tuple[Any, ...], Any]:
    """Flatten a nested dictionary into a flattened dictionary.

    Converts a nested dictionary into a flattened version where the keys 
    are tuples that preserve the structure of the original nested keys.

    For example:
    
        {
            'a': 1,
            'b': {
                'c': 2,
                'd': {
                    'e': 3
                }
            }
        }
    
    would flatten to:

        {
            ('a',): 1,
            ('b','c'): 2,
            ('b','d','e'): 3
        }

    Parameters
    ----------
    dictionary : dict
        Nested dictionary to flatten.

    Returns
    -------
    dict
        Flattened dictionary.

    """
    flattened_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            flattend_value = flatten_dict(value)
            for sub_key, sub_value in flattend_value.items():
                flattened_dict[(key,) + sub_key] = sub_value
        else:
            flattened_dict[(key,)] = value
    return flattened_dict


def collect_settings() -> dict[tuple[Any, ...], Any]:
    """Collect and merge settings from package and root directories.

    Loads default settings YAML from package directory. 
    Checks for package override YAML in package dir.
    Checks for root override YAML in root dir.
    Merges everything into a single flattened settings dict.

    Precedence is:

    root dir overrides > package dir overrides > default

    Returns
    -------
    dict
        Flattened dictionary of collected settings.

    """
    sample_settings_path = PACKAGE_DIRECTORY.joinpath("config", "default_settings.yaml")
    _settings = flatten_dict(open_yaml(sample_settings_path))

    package_settings_path = PACKAGE_DIRECTORY.joinpath(_settings[("package_settings",)])
    if package_settings_path.exists():
        package_settings = flatten_dict(open_yaml(package_settings_path))
        _update_settings(_settings, package_settings)

    root_setting_path = ROOT_DIRECTORT.joinpath(_settings[("local_settings",)])
    if root_setting_path.exists():
        root_settings = flatten_dict(open_yaml(root_setting_path))
        _update_settings(_settings, root_settings)

    return _settings


def _update_settings(_settings, new_settings):
    for key, value in new_settings.items():
        if key in _settings:
            _settings[key] = value


settings = collect_settings()


class DefaultColumns(BaseModel):
    year: str = settings[("columns", "year")]
    household_id: str = settings[("columns", "household_id")]
    commodity_code: str = settings[("columns", "commodity_code")]
    job_code: str = settings[("columns", "job_code")]
    weight: str = settings[("columns", "weight")]


class Defaults(BaseModel):
    online_dir: str = settings[("online_directory",)]
    package_dir: Path = PACKAGE_DIRECTORY
    root_dir: Path = ROOT_DIRECTORT

    if Path(settings[("local_directory",)]).is_absolute():
        local_dir: Path = Path(settings[("local_directory",)])
    elif settings[("in_root",)]:
        local_dir: Path = root_dir.joinpath(settings[("local_directory",)])
    else:
        local_dir: Path = package_dir.joinpath(settings[("local_directory",)])

    archive_files: Path = local_dir.joinpath(settings[("archive_files",)])
    unpacked_data: Path = local_dir.joinpath(settings[("unpacked_data",)])
    extracted_data: Path = local_dir.joinpath(settings[("extracted_data",)])
    cleaned_data: Path = local_dir.joinpath(settings[("cleaned_data",)])
    external_data: Path = local_dir.joinpath(settings[("external_data",)])
    maps: Path = local_dir.joinpath(settings[("maps",)])
    cached_data: Path = local_dir.joinpath(settings[("cached_data",)])

    first_year: int = settings[("first_year",)]
    last_year: int = settings[("last_year",)]

    columns: DefaultColumns = DefaultColumns()

    def model_post_init(self, __contex=None) -> None:
        self.archive_files.mkdir(parents=True, exist_ok=True)
        self.unpacked_data.mkdir(parents=True, exist_ok=True)
        self.extracted_data.mkdir(parents=True, exist_ok=True)
        self.cleaned_data.mkdir(parents=True, exist_ok=True)
        self.external_data.mkdir(parents=True, exist_ok=True)
        self.maps.mkdir(parents=True, exist_ok=True)
        self.cached_data.mkdir(parents=True, exist_ok=True)


class Metadata:
    """
    A dataclass for accessing metadata used in other parts of the project.

    """

    metadata_files = [
        "instruction",
        "tables",
        # "maps",
        # "household",
        # "commodities",
        # "occupations",
        "industries",
        # "schema",
        "other",
        # "external_data",
    ]
    instruction: dict[str, Any]
    tables: dict[str, Any]
    maps: dict[str, Any]
    household: dict[str, Any]
    commodities: dict[str, Any]
    occupations: dict[str, Any]
    industries: dict[str, Any]
    schema: dict[str, Any]
    other: dict[str, Any]
    external_data: dict[str, Any]

    def __init__(self) -> None:
        self.reload()

    def reload(self):
        for file_name in self.metadata_files:
            self.reload_file(file_name)

    def reload_file(self, file_name):
        package_metadata_path = settings[("package_metadata", file_name)]
        local_metadata_path = ROOT_DIRECTORT.joinpath(
            settings[("local_metadata", file_name)]
        )
        interpreter = self.get_interpreter(file_name)
        _metadata: dict = open_yaml(package_metadata_path, interpreter=interpreter)
        interpreter = self.get_interpreter(file_name, _metadata)
        if local_metadata_path.exists():
            local_metadata = open_yaml(local_metadata_path, interpreter=interpreter)
            _metadata.update(local_metadata)
        setattr(self, file_name, _metadata)

    def get_interpreter(
        self, file_name: str, context: dict | None = None
    ) -> Callable[[str], str] | None:
        context = context or {}
        if f"{file_name}_interpreter" in dir(self):
            interpreter = getattr(self, f"{file_name}_interpreter")
            interpreter = functools.partial(interpreter, context=context)
        else:
            interpreter = None
        return interpreter



metadata = Metadata()
defaults = Defaults()
