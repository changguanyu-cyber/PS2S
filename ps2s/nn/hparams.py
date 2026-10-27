from typing import Protocol, Type, TypedDict

class HParamsDict(TypedDict):
    cls: Type

class HasHParams(Protocol):
    hparams: HParamsDict

def from_hparams(hparams: HParamsDict):
    cls = hparams['cls']
    kwargs = {k: v for k, v in hparams.items() if k != 'cls'}
    return cls(**kwargs)
