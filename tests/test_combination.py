# standard library
from dataclasses import dataclass
from typing import Any as Any_


# dependencies
from ndtools import All, Any, Combinable


def eq(left: Any_, right: Any_, /) -> bool:
    return super(type(left), left).__eq__(right)


def test_All() -> None:
    assert eq(All([0]) & 1, All([0, 1]))
    assert eq(All([0]) | 1, Any([All([0]), 1]))
    assert eq(All([0]) & All([1]), All([0, 1]))
    assert eq(All([0]) | All([1]), Any([All([0]), All([1])]))


def test_Any() -> None:
    assert eq(Any([0]) & 1, All([Any([0]), 1]))
    assert eq(Any([0]) | 1, Any([0, 1]))
    assert eq(Any([0]) & Any([1]), All([Any([0]), Any([1])]))
    assert eq(Any([0]) | Any([1]), Any([0, 1]))


def test_Combinable() -> None:
    @dataclass
    class Test(Combinable):
        data: float

    assert eq(Test(0) & 1, All([Test(0), 1]))
    assert eq(Test(0) | 1, Any([Test(0), 1]))
