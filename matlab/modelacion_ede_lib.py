# modelacion_ede_lib.py
# Librería amigable para orquestar ModelacionEDE.

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, Union

import pandas as pd

from ModelacionEDE import ModelacionEDE


# Logger "librería-friendly": no configura el root logger.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SerieModeler:
    """
    Orquestador para ModelacionEDE.
    No modifica ninguna función matemática del modelo subyacente.
    Encapsula el pipeline de pasos y expone una API estable para uso como librería.
    """

    def __init__(
        self,
        dt: float,
        nombre: str,
        frecuencia: str = "días",
        k: int = 1000,
        mostrar_figuras: bool = True,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self.dt = dt
        self.nombre = nombre
        self.frecuencia = frecuencia
        self.k = k
        self.mostrar_figuras = mostrar_figuras
        self.log = logger_ if logger_ is not None else logger

    def _run_pipeline(self, df: pd.DataFrame) -> Tuple[ModelacionEDE, Dict]:
        """
        Ejecuta los pasos 0–7 definidos en ModelacionEDE,
        conservando su implementación original.
        """
        m = ModelacionEDE(self.dt, df, self.nombre, self.frecuencia)

        # Paso 0: descripción
        self.log.info("Paso 0: descripción")
        m.paso0()

        # Paso 1: PAC (criterio de continuidad)
        self.log.info("Paso 1: PAC")
        if not m.paso1():
            return m, {
                "tipo_modelo": "ERROR",
                "parametros": None,
                "error": "No cumple con el requisito de autocorrelación parcial",
                "mape": None,
            }

        # Paso 2: Hurst/Fractal
        self.log.info("Paso 2: Hurst/Fractal")
        m.paso2()

        # Paso 3: retornos
        self.log.info("Paso 3: retornos")
        m.paso3()

        # Paso 4: normalidad
        self.log.info("Paso 4: normalidad")
        m.paso4()

        # Paso 5: homocedasticidad
        self.log.info("Paso 5: homocedasticidad")
        m.paso5()

        # Paso 6: estimación
        self.log.info("Paso 6: estimación")
        m.paso6()

        # Paso 7: pronóstico y evaluación
        self.log.info("Paso 7: pronóstico")
        if m.es_tendencia:
            Xp = m.paso7(self.k)
            mape, _ = m.analizar_pronostico(Xp)
            out = {
                "tipo_modelo": "EDELH",
                "parametros": m.parametros.get("EDELH"),
                "error": None,
                "mape": mape["mean"],
            }
        elif m.es_reg_media:
            X_OU, X_EDEL = m.paso7(self.k)
            if X_OU is None and X_EDEL is None:
                out = {
                    "tipo_modelo": "ERROR",
                    "parametros": None,
                    "error": "Sin modelos válidos",
                    "mape": None,
                }
            elif X_OU is None:
                mape, _ = m.analizar_pronostico(X_EDEL)
                out = {
                    "tipo_modelo": "EDEL",
                    "parametros": m.parametros.get("EDEL"),
                    "error": None,
                    "mape": mape["mean"],
                }
            elif X_EDEL is None:
                mape, _ = m.analizar_pronostico(X_OU)
                out = {
                    "tipo_modelo": "OU",
                    "parametros": m.parametros.get("OU"),
                    "error": None,
                    "mape": mape["mean"],
                }
            else:
                mape_ou, _ = m.analizar_pronostico(X_OU)
                mape_edel, _ = m.analizar_pronostico(X_EDEL)
                if mape_ou["mean"] <= mape_edel["mean"]:
                    out = {
                        "tipo_modelo": "OU",
                        "parametros": m.parametros.get("OU"),
                        "error": None,
                        "mape": mape_ou["mean"],
                    }
                else:
                    out = {
                        "tipo_modelo": "EDEL",
                        "parametros": m.parametros.get("EDEL"),
                        "error": None,
                        "mape": mape_edel["mean"],
                    }
        else:
            out = {
                "tipo_modelo": "ERROR",
                "parametros": None,
                "error": "Tipo de proceso no identificado",
                "mape": None,
            }

        if self.mostrar_figuras and hasattr(m, "mostrar_figuras"):
            try:
                m.mostrar_figuras()
            except Exception:
                # La librería no debe fallar por rendering de figuras
                pass

        return m, out

    def procesar(self, df: Union[pd.Series, pd.DataFrame]) -> Dict:
        """
        Ejecuta todo el pipeline sobre una serie (DataFrame 1-col o Series).
        """
        if isinstance(df, pd.Series):
            df = df.to_frame()
        if not isinstance(df, pd.DataFrame) or df.shape[1] != 1:
            raise ValueError(
                "df debe ser un DataFrame unidimensional (una sola columna) o una Series."
            )

        _, result = self._run_pipeline(df)
        return result

__all__ = ["SerieModeler"]
