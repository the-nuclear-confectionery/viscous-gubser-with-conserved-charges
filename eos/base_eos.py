from abc import ABC, abstractmethod
from typing import Union
import numpy as np

class BaseEoS(ABC):
    @abstractmethod
    def pressure(
        self,
        T: Union[float, np.ndarray],
        mu: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        pass

    @abstractmethod
    def energy(
        self,
        T: Union[float, np.ndarray],
        mu: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        pass

    @abstractmethod
    def entropy(
        self,
        T: Union[float, np.ndarray],
        mu: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        pass

    @abstractmethod
    def number(
        self,
        T: Union[float, np.ndarray],
        mu: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        pass