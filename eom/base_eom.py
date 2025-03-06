from abc import ABC, abstractmethod
from typing import Union
import numpy as np

class BaseEoM(ABC):
    @abstractmethod
    def dT_drho(self, ys: np.ndarray, rho: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        pass

    @abstractmethod
    def dmu_drho(self, ys: np.ndarray, rho: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        pass

    @abstractmethod
    def dpi_drho(self, ys: np.ndarray, rho: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        pass

    def eom(self, ys: np.ndarray, rho: Union[float, np.ndarray], ideal_evolution: bool = False) -> np.ndarray:
        dT = self.dT_drho(ys, rho)
        dmu = self.dmu_drho(ys, rho)
        dpi = self.dpi_drho(ys, rho) if not ideal_evolution else 0
        return np.array([dT, *np.array(dmu).reshape(-1,), dpi])