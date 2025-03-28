from typing import Union, Optional, Dict
from eom.base_eom import BaseEoM
import numpy as np

def get_eom(
    eos_type: str,
    eos_params: Optional[Dict[str, Union[float, np.ndarray]]] = None,
    eom_params: Optional[Dict[str, float]] = None
) -> BaseEoM:
    """
    Returns an instance of a BaseEoM implementation.

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
    An instance of BaseEoM.
    """
    if eos_type.lower() == "eos1":
        from eos.massless_qgp_eos import MasslessQGPEoS
        from eom.massless_qgp_eom import MasslessQGPEoM
        # Create an EOS instance (the massless QGP EOS does not need reference scales)
        eos_instance = MasslessQGPEoS()
        return MasslessQGPEoM(eos_instance, eom_params=eom_params)
    elif eos_type.lower() == "eos2":
        if eos_params is None:
            raise ValueError("EoS params must be provided for EoS2.")
        from eos.conformal_plasma_eos import ConformalPlasmaEoS
        from eom.conformal_plasma_eom import ConformalPlasmaEoM
        eos_instance = ConformalPlasmaEoS(eos_params=eos_params)
        return ConformalPlasmaEoM(eos_instance, eos_params=eos_params, eom_params=eom_params)
    else:
        raise ValueError(f"Unknown EoS of type {eos_type}")