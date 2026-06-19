from typing import Optional

from capymoa.drift.base_detector import MOADriftDetector

from moa.classifiers.core.driftdetection import EDDM as _EDDM


class EDDM(MOADriftDetector):
    """Drift-Detection-Method (DDM) Drift Detector

    Example:
    --------

    >>> import numpy as np
    >>> from capymoa.drift.detectors import EDDM
    >>> np.random.seed(0)
    >>>
    >>> detector = EDDM()
    >>>
    >>> data_stream = np.random.randint(2, size=2000)
    >>> for i in range(999, 2000):
    ...     data_stream[i] = np.random.randint(4, high=8)
    >>>
    >>> for i in range(2000):
    ...     detector.add_element(data_stream[i])
    ...     if detector.detected_change():
    ...         print('Change detected in data: ' + str(data_stream[i]) + ' - at index: ' + str(i))
    Change detected in data: 4 - at index: 1005

    Reference:
    ----------

    Early Drift Detection Method. Manuel Baena-Garcia, Jose Del Campo-Avila,
    Raul Fidalgo, Albert Bifet, Ricard Gavalda, Rafael Morales-Bueno. In Fourth
    International Workshop on Knowledge Discovery from Data Streams, 2006.

    """

    def __init__(
        self,
        min_n_errors: int = 30,
        warning_level: float = 0.95,
        out_control_level: float = 0.9,
        CLI: Optional[str] = None,
    ):
        if CLI is None:
            CLI = f"-A {min_n_errors} -C {warning_level} -B {out_control_level}"

        super().__init__(moa_detector=_EDDM(), CLI=CLI)

        self.min_n_errors = min_n_errors
        self.warning_level = warning_level
        self.out_control_level = out_control_level
        self.get_params()