import numpy as np
from typing import Union, Optional, Dict
from eos.massless_qgp_eos import MasslessQGPEoS
from eos.conformal_plasma_eos import ConformalPlasmaEoS
from eos.base_eos import BaseEoS

def get_eos(
    eos_type: str,
    eos_params: Optional[Dict[str, Union[float, np.ndarray]]] = None
) -> BaseEoS:
    """
    Returns an instance of a BaseEoS implementation.

    Parameters:
    -----------
    eos_type: str
        For example, "EoS1" or "EoS2".
    eos_params: dict, optional
        A dictionary that contains EoS-specific parameters. For example,
        for the EoS2, the dictionary should contain:
          - T_ast: Temperature scale parameter (float)
          - mu_ast: Chemical potential scale parameter (float)

    Returns:
    --------
    An instance of BaseEoS.
    """
    if eos_type.lower() == "eos1":
        return MasslessQGPEoS()
    elif eos_type.lower() == "eos2":
        if eos_params is None:
            raise ValueError("EoS params must be provided for EoS2.")
        return ConformalPlasmaEoS(eos_params=eos_params)
    else:
        raise ValueError(f"Unknown eos_type: {eos_type}")