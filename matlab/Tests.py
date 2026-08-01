import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import het_arch, lilliefors
from statsmodels.tsa.stattools import pacf, acovf
from statsmodels.graphics.tsaplots import plot_pacf
import statsmodels.api as sm
from scipy.stats import chi2
import pandas as pd
from typing import Union

# =============================================================================
# Función para pruebas de normalidad
# =============================================================================
def test_normality(data):
    """
    Realiza pruebas de normalidad sobre los datos y determina si siguen una distribución normal.

    Parameters
    ----------
        data : numpy.ndarray
            Serie de datos unidimensional a analizar.

    Returns
    -------
        bool : bool
            True si los datos son normales, False en caso contrario.
    """
    print("=== Pruebas de Normalidad ===")
    data = (data - np.mean(data))/np.std(data)

    # Realizar todas las pruebas estadísticas
    shapiro_stat, shapiro_p = stats.shapiro(data)
    jb_stat, jb_p = stats.jarque_bera(data)
    ad_result = stats.anderson(data, dist='norm')
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(0, 1))
    lilliefors_stat, lilliefors_p = lilliefors(data)

    # Verificar si al menos un test no rechaza la hipótesis nula (p > 0.05)
    p_values = [shapiro_p, jb_p, ks_p, lilliefors_p]
    if any(p > 0.05 for p in p_values):
        print("Al menos un estadístico no rechaza la hipótesis nula")
        es_normal = True
    else:
        # Mostrar gráfico QQ-plot
        fig = plt.figure()
        sm.qqplot(data, line='45')
        plt.title('QQ-Plot de los datos')
        plt.close(fig)  # Cerramos la figura para que no se muestre

    # Imprimir resultados detallados
    print("\nResultados detallados de las pruebas:")
    print(f"Shapiro-Wilk: statistic = {shapiro_stat:.5f}, p-value = {shapiro_p:.5f}")
    print(f"Jarque-Bera: statistic = {jb_stat:.5f}, p-value = {jb_p:.5f}")
    print(f"Anderson-Darling: statistic = {ad_result.statistic:.5f}")
    for sig, crit in zip(ad_result.significance_level, ad_result.critical_values):
        print(f"  {sig}%: critical value = {crit:.5f}")
    print(f"Kolmogorov-Smirnov: statistic = {ks_stat:.5f}, p-value = {ks_p:.5f}")
    print(f"Lilliefors: statistic = {lilliefors_stat:.5f}, p-value = {lilliefors_p:.5f}")

    return es_normal

# =============================================================================
# Función para pruebas de bondad de ajuste
# =============================================================================
def test_goodness_of_fit(data: np.ndarray, alpha: float = 0.05) -> bool:
    """
    Realiza una prueba de bondad de ajuste chi-cuadrado para verificar si los datos
    siguen una distribución uniforme.

    Parameters
    ----------
    data : numpy.ndarray
        Serie de datos a analizar.
    alpha : float, optional
        Nivel de significancia para la prueba. Por defecto es 0.05.

    Returns
    -------
    bool
        True si los datos pasan la prueba (p-valor > alpha),
        False si se rechaza la hipótesis nula (p-valor <= alpha).
    """
    print("=== Prueba de Bondad de Ajuste ===")
    chi2_stat, chi2_p = stats.chisquare(data)
    print(f"Chi-cuadrado: statistic = {chi2_stat:.5f}, p-value = {chi2_p:.5f}\n")

    return chi2_p > alpha

# =============================================================================
# Función para pruebas de heterocedasticidad condicional
# =============================================================================
def test_heteroscedasticity(data: np.ndarray, alpha: float = 0.05) -> bool:
    """
    Realiza la prueba de heterocedasticidad condicional de Engle (ARCH test)
    para verificar si los datos presentan heterocedasticidad.

    Parameters
    ----------
    data : numpy.ndarray
        Serie de datos a analizar.
    alpha : float, optional
        Nivel de significancia para la prueba. Por defecto es 0.05.

    Returns
    -------
    bool
        True si los datos son homocedásticos (p-valor > alpha),
        False si se detecta heterocedasticidad (p-valor <= alpha).
    """
    print("=== Prueba de heterocedasticidad ===\n")

    # Engle's ARCH Test
    stat, p, _, _ = het_arch(data)
    print(f"Engle ARCH Test: Estadística={stat:.5f}, p-valor={p:.5f}\n")

    return p > alpha

# =============================================================================
# Exponente de Hurst H(1)
# =============================================================================
def exp_hurst(S: np.ndarray) -> float:
    """
    Calcula el exponente de Hurst de orden 1, definido en términos del comportamiento
    asintótico del rango reescalado en función del lapso de tiempo de una serie temporal.

    La relación asintótica es: |x(t+r) - x(t)| / x(t) ~ r^H

    Parameters
    ----------
    S : numpy.ndarray
        Serie temporal unidimensional (recomendado T > 50)

    Returns
    -------
    float
        Coeficiente de Hurst H

    Raises
    ------
    ValueError
        Si la serie temporal no es unidimensional o tiene menos de 50 datos
    """
    # Validación de entrada
    if S.ndim > 1:
        raise ValueError('La serie temporal debe ser unidimensional')
    if len(S) < 50:
        raise ValueError('La serie temporal debe tener al menos 50 datos')

    # Parámetros del cálculo
    maxT = 19  # cantidad máxima de ventanas escaladas
    L = len(S)
    H_values = []

    # Calcular H para diferentes ventanas
    for Tmax in range(5, maxT + 1):
        # Preparar datos para la regresión
        x = np.arange(1, Tmax + 1)
        mcord = np.zeros(Tmax)

        # Calcular diferencias y valores para cada lag
        for tt in range(1, Tmax + 1):
            # Calcular diferencias y valores base
            dV = S[tt:L:tt] - S[0:L-tt:tt]
            VV = S[0:L:tt] # Y = VV
            N = len(dV) + 1
            X = np.arange(1, N + 1)

            # Calcular coeficientes de regresión
            mx = np.mean(X)
            my = np.mean(VV)
            SSxx = np.sum(X**2) - N * mx**2
            SSxy = np.sum(X * VV) - N * mx * my

            # Calcular pendiente e intercepto
            slope = SSxy / SSxx
            intercept = my - slope * mx

            # Calcular valores ajustados y residuos
            ddVd = dV - slope
            VVVd = VV - slope * X - intercept

            # Calcular razón de medias
            mcord[tt-1] = np.mean(np.abs(ddVd)) / np.mean(np.abs(VVVd))

        # Calcular H para esta ventana
        mx = np.mean(np.log10(x))
        my = np.mean(np.log10(mcord))
        SSxx = np.sum(np.log10(x)**2) - Tmax * mx**2
        SSxy = np.sum(np.log10(x) * np.log10(mcord)) - Tmax * mx * my

        H_values.append(SSxy / SSxx)

    # Retornar el promedio de los valores de H
    return np.mean(H_values)



# =============================================================================
# Función para calcular exponente de Hurst y dimensión fractal
# =============================================================================
def fractalvol(data):
    """
    Calcula la dimensión fractal y la desviación estándar del estimador (volatilidad fractal)
    de una serie de tiempo.

    Parameters
    ----------
    data : Union[pandas.DataFrame, numpy.ndarray]
        Serie de tiempo a analizar. Puede ser un DataFrame de pandas con una sola columna
        o un array de NumPy unidimensional.

    Returns
    -------
    tuple
        dimension : float
            Dimensión fractal estimada (negativo de la pendiente de la regresión log-log)
        standard_dev : float
            Desviación estándar de la pendiente según la teoría de OLS.

    Raises
    ------
    ValueError
        Si los datos no tienen el formato correcto o si hay problemas con los datos.
    """
    # Convertir entrada a array de NumPy
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("El DataFrame debe tener exactamente una columna")
        y = data.iloc[:, 0].to_numpy()
        x = np.arange(len(y))
    elif isinstance(data, np.ndarray):
        if data.ndim > 1:
            raise ValueError("El array debe ser unidimensional")
        y = data
        x = np.arange(len(y))
    else:
        raise TypeError("Los datos deben ser un DataFrame de pandas o un array de NumPy")

    # Validar datos
    if len(y) < 10:
        raise ValueError("Se requieren al menos 10 puntos para el cálculo de la dimensión fractal")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Los datos no pueden contener valores NaN o Inf")

    # Normalizar los datos
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    # Calcular el ancho mínimo de los intervalos
    dx = np.diff(x_norm)
    minwidth = int(np.abs(np.ceil(np.log2(np.min(dx)))) - 1)
    n_values = np.zeros(minwidth)
    scales = 2.0 ** (-np.arange(1, minwidth + 1))

    # Ciclo de box-counting para cada escala j = 1, 2, ..., minwidth
    for j in range(minwidth):
        width = scales[j]
        x_positions = np.arange(0, 1 + width, width)

        for x_pos in x_positions:
            # Encontrar puntos en el intervalo actual
            mask = (x_norm >= x_pos) & (x_norm < x_pos + width)
            if np.isclose(x_pos + width, 1, atol=1e-6):
                mask[-1] = True

            y_values = y_norm[mask]

            if len(y_values) > 0:
                if len(y_values) == 1:
                    n_values[j] += 1
                else:
                    rawcount = (np.max(y_values) - np.min(y_values)) / width
                    rawcount += np.remainder(np.min(y_values), width)
                    n_values[j] += np.ceil(rawcount)

    # Calcular la regresión log-log
    x_log = np.log(scales)
    y_log = np.log(n_values)

    # Calcular gradiente local y eliminar outliers
    s_all = -np.gradient(y_log) / np.gradient(x_log)
    IQR = np.subtract(*np.percentile(s_all, [75, 25]))
    median_s = np.median(s_all)
    valid = np.abs(s_all - median_s) <= IQR / 2

    # Regresión lineal OLS: log(n_values) = A + B log(r)
    X = np.vstack([np.ones_like(x_log[valid]), x_log[valid]]).T
    y = y_log[valid]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    dimension = -beta[1] # La dimensión fractal se define como -B (donde B es la pendiente)

    # Calcular desviación estándar
    residuals = y - X @ beta
    C = np.linalg.inv(X.T @ X)
    sigma2 = (residuals.T @ residuals) * C
    standard_dev = np.sqrt(sigma2[1, 1])

    return dimension, standard_dev


# =============================================================================
# Función para pruebas de autocorrelación parcial y autocovarianza
# =============================================================================
def test_autocorrelation(data: np.ndarray, nombre: str, nlags: int = 20, conf_level: float = 0.9973) -> tuple[bool, float]:
    """
    Realiza pruebas de autocorrelación parcial y verifica la significancia estadística
    de los coeficientes PACF.

    Parameters
    ----------
        data : numpy.ndarray
            Serie de datos unidimensional a analizar.
        nombre : str
            Nombre de la serie para el título del gráfico.
        nlags : int, optional
            Número de lags a considerar en el análisis. Por defecto es 20.
        conf_level : float, optional
            Nivel de confianza para los intervalos de confianza. Por defecto es 0.9973.

    Returns
    -------
        pacf_vals : numpy.ndarray
            Valores PACF de los primeros nlags.
        conf_interval : float
            Tamaño del intervalo de confianza para determinar autocorrelaciones paricales significativas.
            z_{(1+alpha) / 2} / sqrt(len(data))
    """
    print("=== Prueba de Autocorrelación Parcial ===\n")

    # Calcular autocorrelación parcial
    pacf_vals = pacf(data, nlags=nlags)

    # Calcular intervalos de confianza
    n = len(data)
    z_score = stats.norm.ppf((1 + conf_level) / 2)
    conf_interval = z_score / np.sqrt(n)

    # Resultados
    print(f"Nivel de confianza: {conf_level*100:.2f}%")
    print(f"Intervalo de confianza: ±{conf_interval:.5f}")
    print(f"\nValores PACF para los primeros {nlags} lags:")
    for i in range(len(pacf_vals)):
        print(f"  Lag {i}: {pacf_vals[i]:.5f}")

    # Plot PACF
    fig = plt.figure()
    plot_pacf(data, lags=nlags, alpha=1-conf_level, title=f'Función de Autocorrelación Parcial (PACF) - {nombre}')
    plt.xlabel('Lags')
    plt.ylabel('PACF')
    plt.grid(True)
    plt.close(fig)  # Cerramos la figura para que no se muestre

    return pacf_vals, conf_interval

def white_test(R, alpha: float = 0.05):
    """
    Realiza el test de White para verificar la heterocedasticidad en la serie R.

    El test de White es una prueba de heterocedasticidad que verifica si la varianza
    de los residuos es constante. Se basa en una regresión auxiliar de los residuos
    al cuadrado sobre las variables independientes y sus términos no lineales.

    Parameters
    ----------
    R : numpy.ndarray
        Vector de residuos de la regresión principal (1D).

    Returns
    -------
    bool
        True si los datos pasan la prueba (p-valor > alpha),
        False si se rechaza la hipótesis nula (p-valor <= alpha).

    Notes
    -----
    El test se realiza en tres pasos:
    1. Regresión original de R sobre el tiempo y su cuadrado
    2. Regresión auxiliar de los residuos al cuadrado
    3. Cálculo del estadístico chi-cuadrado y su p-valor
    """
    # Validación de entrada
    R = np.asarray(R)
    if R.ndim != 1:
        raise ValueError("R debe ser un vector unidimensional")

    T = len(R)
    if T < 5:
        raise ValueError("Se requieren al menos 5 observaciones para el test")

    # a) Regresión auxiliar con términos no lineales
    t = np.arange(1, T + 1)
    X1 = np.column_stack((t, t**2))  # Variables independientes
    X = sm.add_constant(X1)  # Añadir término constante

    # Regresión original
    model = sm.OLS(R, X)
    results = model.fit()
    resid = results.resid

    # b) Regresión auxiliar con los residuos al cuadrado
    X_white = np.column_stack((X1, X1[:, 1]**2))  # Añadimos el término cuadrado del tiempo
    model_aux = sm.OLS(resid**2, X_white)   # Regresión de los residuos al cuadrado
    results_aux = model_aux.fit()

    # c) Estadístico de prueba
    R2 = 1 - np.sum(results_aux.resid**2) / np.sum((resid**2 - np.mean(resid**2))**2)
    chi2_stat = T * R2 # Estadístico de prueba

    # p-valor para el test de White
    pval = 1 - chi2.cdf(chi2_stat, df=X_white.shape[1] - 1)

    return pval > alpha
