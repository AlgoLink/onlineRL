from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    SupportsFloat,
    Tuple,
    Union,
    overload,
)

import re
import importlib


ENV_ID_RE = re.compile(
    r"^(?:(?P<namespace>[\w:-]+)\/)?(?:(?P<name>[\w:.-]+?))(?:-v(?P<version>\d+))?$"
)

def load(name: str) -> callable:
    """Loads an environment with name and returns an environment creation function
    Args:
        name: The environment name
    Returns:
        Calls the environment constructor
    """
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


def parse_env_id(id: str) -> Tuple[Optional[str], str, Optional[int]]:
    """Parse environment ID string format.
    This format is true today, but it's *not* an official spec.
    [namespace/](env-name)-v(version)    env-name is group 1, version is group 2
    2016-10-31: We're experimentally expanding the environment ID format
    to include an optional namespace.
    ex, parse_env_id("namespace/envname-v12")
    Args:
        id: The environment id to parse
    Returns:
        A tuple of environment namespace, environment name and version number
    Raises:
        Error: If the environment id does not a valid environment regex
    """
    match = ENV_ID_RE.fullmatch(id)
    if not match:
        raise error.Error(
            f"Malformed environment ID: {id}."
            f"(Currently all IDs must be of the form [namespace/](env-name)-v(version). (namespace is optional))"
        )
    namespace, name, version = match.group("namespace", "name", "version")
    if version is not None:
        version = int(version)

    return namespace, name, version

def get_env_id(ns: Optional[str], name: str, version: Optional[int]) -> str:
    """Get the full env ID given a name and (optional) version and namespace. Inverse of :meth:`parse_env_id`.
    ex,get_env_id("filename","envname","1")
    Args:
        ns: The environment namespace
        name: The environment name
        version: The environment version
    Returns:
        The environment id
    """

    full_name = name
    if version is not None:
        full_name += f"-v{version}"
    if ns is not None:
        full_name = ns + "/" + full_name
    return full_name