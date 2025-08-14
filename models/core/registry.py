# models.py
from __future__ import annotations

from typing import Iterable, Callable

import numpy as np

np.float_ = np.float64


class RoleRegistry:
    """
    Maps role -> set(classes). Lets us declare/inspect 'input', 'output',
    'calc', or any arbitrary role like 'x'.
    """

    _by_role: dict[str, set[type]] = {}

    @classmethod
    def register(cls, role: str, typ: type) -> None:
        role = role.lower()
        cls._by_role.setdefault(role, set()).add(typ)

    @classmethod
    def classes_for(cls, role: str) -> set[type]:
        return cls._by_role.get(role.lower(), set())

    @classmethod
    def all_roles(cls) -> Iterable[str]:
        return cls._by_role.keys()


def register_role(role: str) -> Callable[[type], type]:
    """
    Decorator: @register_role("input") on a class sets _role and registers it.
    """
    role = role.lower()

    def _wrap(typ: type) -> type:
        setattr(typ, "_role", role)
        RoleRegistry.register(role, typ)
        return typ

    return _wrap
