import numpy as np
from Tests import *
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats

class ModelacionEDE:
    def __init__(self, dt, X, name, frecuencia="días"):
        """
        Inicializa la clase ModelacionEDE.

        Parameters
        ----------
        dt : float
            Intervalo de tiempo entre observaciones.
        X : pandas.DataFrame
            DataFrame con los datos de la serie temporal.
        name : str
            Nombre de la serie temporal.
        frecuencia : str, optional
            Unidad de tiempo del muestreo (ej: "días", "horas", "semanas"). Por defecto es "días".
        """
        self.dt = dt
        self.X = X.iloc[:, 0].values  # Almacenamos directamente los valores
        self.name = name
        self.frecuencia = frecuencia
        self.figuras = []  # Lista para almacenar las figuras

    def guardar_figura(self, fig):
        """
        Guarda una figura para mostrarla posteriormente.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figura a guardar.
        """
        self.figuras.append(fig)

    def mostrar_figuras(self):
        """
        Muestra todas las figuras guardadas.
        """
        for fig in self.figuras:
            plt.figure(fig.number)
            plt.show()

    ##############################################
    # Paso 0: Descripción estadística del activo #
    ##############################################
    def paso0(self) -> None:
        """
        Realiza un análisis estadístico descriptivo básico de los datos.
        Calcula estadísticas básicas y genera un histograma de la serie.
        """
        values = self.X

        # Calcular estadísticas
        estadisticas = {
            'Mínimo': np.min(values),
            'Máximo': np.max(values),
            'Media': np.mean(values),
            'Mediana': np.median(values),
            'Varianza': np.var(values),
            'Desviación Estándar': np.std(values),
            'Asimetría': stats.skew(values),
            'Curtosis': stats.kurtosis(values)
        }

        # Imprimir estadísticas
        print("\n=== Estadísticas Descriptivas ===")
        for estadistico, value in estadisticas.items():
            print(f"{estadistico}: {value:.5f}")

        # Generar histograma
        fig = plt.figure(figsize=(10, 6))
        plt.hist(values, bins='auto', density=True, alpha=0.7)
        plt.title(f'Histograma de {self.name}')
        plt.xlabel(f'{self.name}')
        plt.ylabel('Densidad')
        plt.grid(True, alpha=0.3)
        self.guardar_figura(fig)

    ###################################
    # Paso 1: Autocorrelación parcial #
    ###################################
    def paso1(self, conf_level = 0.9973) -> bool:
        """
        Verifica si la serie temporal tiene una dependencia significativa
        solo con su primer rezago (PACF(1) significativo).

        Parameters
        ----------
        conf_level : float, optional
            Nivel de confianza para los intervalos de confianza. Por defecto es 0.9973.

        Returns
        -------
        bool
            True si la serie solo depende significativamente de su primer rezago,
            False en caso contrario.
        """
        # Guardar la figura actual
        current_fig = plt.gcf()
        if current_fig.get_size_inches().prod() > 0:
            self.guardar_figura(current_fig)

        pacf_values, conf_interval = test_autocorrelation(self.X, self.name, conf_level=conf_level)

        # Verificar significancia de PACF(1)
        self.rho = pacf_values[1]
        pacf_1 = abs(self.rho) > conf_interval

        # Verificar si hay otros lags significativos
        other_lags_significant = [abs(pacf_values[i]) > conf_interval for i in range(2, len(pacf_values))]

        # Mostrar resultados
        print(f"\nPACF(1) = {self.rho:.5f}")
        print(f"¿PACF(1) es significativo? {'Sí' if pacf_1 else 'No'}")

        if any(other_lags_significant):
            print("\nAdvertencia: Se detectaron otros lags significativos además del lag 1")
            print("Lags significativos:", end=" ")
            significant_lags = np.where(other_lags_significant)[0] + 2
            print(", ".join(map(str, significant_lags)))

        return pacf_1

    ##################################################
    # Paso 2: Dimension fractal y exponente de Hurst #
    ##################################################
    def paso2(self) -> None:
        """
        Calcula la dimensión fractal y el exponente de Hurst para determinar
        el tipo de modelo más apropiado para la serie temporal.

        Interpretación:
        - Si Df = 1.5: Caminata aleatoria
        - Si 1 <= Df < 1.5: Los datos tienen tendencia
        - Si Df >= 1.5: Regresión a la media

        Attributes
        ----------
        es_reg_media : bool
            True si los datos muestran regresión a la media
        es_tendencia : bool
            True si los datos muestran tendencia
        """
        # Calcular dimension fractal y exponente de Hurst
        DF, _ = fractalvol(self.X)
        EH = exp_hurst(self.X)

        print("\n=== Análisis de Dimensión Fractal y Exponente de Hurst ===")
        print(f"Dimensión Fractal (DF) = {DF:.5f}")
        print(f"Exponente de Hurst (EH) = {EH:.5f}")
        print(f"Suma (DF + EH) = {DF + EH:.5f}")

        # Determinar el tipo de modelo
        self.es_reg_media = DF >= 1.5
        self.es_tendencia = 1 <= DF < 1.5

    #################################################
    # Paso 3: Análisis de los retornos instantáneos #
    #################################################
    def retornos(self):
        X_values = self.X
        Rti_OU = X_values[1:] - X_values[:-1]
        Rti_H = (Rti_OU)/X_values[:-1]  # EDELH y EDEL

        return Rti_OU, Rti_H

    def auxiliar_paso3(self, Rti, name_proceso):
        '''
        Analiza los retornos instantáneos y guarda los outliers identificados.

        Parameters
        ----------
        Rti : numpy.ndarray
            Serie de retornos instantáneos a analizar
        name_proceso : str
            Nombre del proceso para identificar los outliers en la memoria
        '''
        test_autocorrelation(Rti, f'Retornos instantáneos {name_proceso}')
        test_heteroscedasticity(Rti)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        # 1) Gráfica de los retornos instantáneos
        axes[0].plot(Rti)
        axes[0].set_xlabel(f'Tiempo en {self.frecuencia}')
        axes[0].set_ylabel('Retornos')
        axes[0].set_title(f'Retornos instantáneos {name_proceso}')

        # 2) Histograma ajustado a una normal
        axes[1].hist(Rti, density=True, alpha=0.6, color='C1')
        mu, sigma = np.mean(Rti), np.std(Rti)
        xmin, xmax = axes[1].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, sigma)
        axes[1].plot(x, p, 'r', linewidth=2)
        axes[1].set_xlabel(f'Retornos instantáneos {name_proceso}')
        axes[1].set_ylabel('Densidad')
        axes[1].set_title('Histograma con Dist. Normal Ajustada')

        # 3) Boxplot para detectar outliers
        box = axes[2].boxplot(Rti)
        axes[2].set_xlabel(f'Retornos instantáneos {name_proceso}')
        axes[2].set_title('Boxplot de retornos')

        # Guardar información de outliers del boxplot
        outliers = box['fliers'][0].get_ydata()
        if len(outliers) > 0:
            # Encontrar los índices de los outliers
            indices_outliers = np.where(np.isin(Rti, outliers))
            # Guardar en la memoria del objeto
            setattr(self, f'outliers_{name_proceso}', {
                'indices': indices_outliers,
            })
            print(f"\nInformación de outliers para {name_proceso}:")
            print(f"Número de outliers: {len(outliers)}")
            print(f"Porcentaje de outliers: {(len(outliers)/len(Rti))*100:.2f}%")
            print(f"Valores de outliers: {outliers}")

        plt.tight_layout()
        self.guardar_figura(fig)

    def paso3(self):
        """
        Analiza los retornos instantáneos según el tipo de modelo determinado en el paso 2.
        Si es_tendencia es True, analiza solo los retornos del modelo EDELH.
        Si es_reg_media es True, analiza los retornos de los modelos EDEL y OU.
        """
        # Obtener los retornos base
        self.RtiOU, self.RtiH = self.retornos()

        if self.es_tendencia:
            print("\n=== Análisis de retornos para modelo EDELH ===")
            self.auxiliar_paso3(self.RtiH, "EDELH")
        elif self.es_reg_media:
            print("\n=== Análisis de retornos para modelo EDEL ===")
            self.auxiliar_paso3(self.RtiH, "EDEL")
            print("\n=== Análisis de retornos para modelo OU ===")
            self.auxiliar_paso3(self.RtiOU, "OU")
        else:
            print("\nAdvertencia: No se ha determinado el tipo de modelo en el paso 2")

    ###################################################
    # Paso 4: Pruebas de normalidad para los retornos #
    ###################################################
    def auxiliar_paso4(self, data, name):
        print(f"\n=== Prueba de normalidad para {name} ===")
        # Guardar la figura actual
        current_fig = plt.gcf()
        if current_fig.get_size_inches().prod() > 0:
            self.guardar_figura(current_fig)

        es_normal = test_normality(data)

        if not es_normal:
            print(f"\nLos datos de {name} no son normales. Intentando eliminar outliers...")

            # Obtener índices de outliers si existen
            outliers_key = f'outliers_{name}'
            if hasattr(self, outliers_key):
                indices_outliers = getattr(self, outliers_key)['indices']
                # Eliminar el 5% de los outliers más extremos
                n_outliers = max(1, int(len(indices_outliers) * 0.05))

                # Calcular cuántos outliers tomar de cada lado
                n_from_start = n_outliers // 2
                n_from_end = n_outliers - n_from_start

                # Tomar los primeros n_from_start y últimos n_from_end outliers
                indices_to_remove = np.concatenate([
                    indices_outliers[:n_from_start],
                    indices_outliers[-n_from_end:]
                ])

                # Crear copia de los datos sin outliers
                data_clean = np.delete(data, indices_to_remove)
                print(f"Se eliminaron {n_outliers} outliers. Probando normalidad nuevamente...")

                # Probar normalidad con datos limpios
                es_normal = test_normality(data_clean)

                if not es_normal:
                    print(f"\nLos datos de {name} aún no son normales después de eliminar outliers.")
                    # Guardar la figura actual antes de mostrar el QQ-plot
                    current_fig = plt.gcf()
                    if current_fig.get_size_inches().prod() > 0:
                        self.guardar_figura(current_fig)
                    plt.show()
                    while True:
                        respuesta = input(f"¿Considera que los datos de {name} siguen una distribución normal? (si/no): ").lower()
                        if respuesta in ['si', 'no']:
                            es_normal = (respuesta == 'si')
                            break
                        print("Por favor, responda 'si' o 'no'")

        return es_normal

    def paso4(self):
        """
        Realiza pruebas de normalidad para los retornos según el tipo de modelo.
        Si los datos no son normales, intenta eliminar outliers y vuelve a probar.
        Si aún así no son normales, solicita decisión manual al usuario.
        """
        if self.es_tendencia:
            self.es_normal_EDELH = self.auxiliar_paso4(self.RtiH, "EDELH")
        elif self.es_reg_media:
            self.es_normal_EDEL = self.auxiliar_paso4(self.RtiH, "EDEL")
            self.es_normal_OU = self.auxiliar_paso4(self.RtiOU, "OU")
        else:
            print("\nAdvertencia: No se ha determinado el tipo de modelo en el paso 2")

    ###########################################
    # Paso 5: Prueba de volatilidad constante #
    ###########################################
    def auxiliar_paso5(self, name_proceso):
        """
        Calcula los residuos según el tipo de modelo y realiza las pruebas de homocedasticidad.

        Parameters
        ----------
        name_proceso : str
            Nombre del proceso ('EDELH', 'EDEL' o 'OU')

        Returns
        -------
        tuple
            (residuos, residuos_cuadrado, es_homocedastico)
        """
        # Calcular residuos según el tipo de modelo
        if name_proceso == 'EDELH':
            residuos = self.RtiH - np.mean(self.RtiH)
        elif name_proceso == 'OU':
            dif = self.X[1:] - self.rho * self.X[:-1]
            residuos = dif - np.mean(dif)
        else:  # EDEL
            dif = self.X[1:] - self.rho * self.X[:-1]
            residuos = (dif - np.mean(dif)) / self.X[:-1]

        # Calcular residuos al cuadrado
        residuos_cuadrado = residuos**2

        # Realizar los tres tests

        # 1. Test de heterocedasticidad
        out_test_arch = test_heteroscedasticity(residuos)

        # 2. Test de autocorrelación para residuos al cuadrado
        pacf_vals, conf_interval = test_autocorrelation(residuos_cuadrado, f'Residuos al cuadrado {name_proceso}')
        hay_autocorr = any(abs(pacf_vals[1:]) > conf_interval)  # Ignoramos el lag 0

        # 3. Test de White
        out_test_white = white_test(residuos)

        # Determinar si es homocedástico (pasa si al menos un test no rechaza H0)
        es_homocedastico = (out_test_arch) or (not hay_autocorr) or (out_test_white)

        # Si ningún test pasa, calcular volatilidad acumulada
        if not es_homocedastico:
            print("\nNingún test pasó. Calculando volatilidad acumulada...")
            N = len(residuos)
            VEL = np.zeros(N)
            for i in range(1, N + 1):
                VEL[i-1] = np.std(residuos[:i])
            VEL = VEL / np.sqrt(self.dt)

            # Graficar volatilidad acumulada
            fig = plt.figure(figsize=(10, 6))
            plt.plot(np.arange(1, N + 1), VEL)
            plt.xlabel(f'Tiempo en {self.frecuencia}')
            plt.ylabel('Volatilidad')
            plt.title(f'Volatilidad acumulada - {name_proceso}')
            plt.grid(True)
            self.guardar_figura(fig)

        return es_homocedastico

    def paso5(self):
        """
        Realiza las pruebas de homocedasticidad para los residuos según el tipo de modelo.
        Si es_tendencia, prueba solo EDELH.
        Si es_reg_media, prueba EDEL y OU.
        """
        if self.es_tendencia:
            print("\n=== Análisis de homocedasticidad para modelo EDELH ===")
            self.es_homocedastico_EDELH = self.auxiliar_paso5('EDELH')
        elif self.es_reg_media:
            print("\n=== Análisis de homocedasticidad para modelo EDEL ===")
            self.es_homocedastico_EDEL = self.auxiliar_paso5('EDEL')
            print("\n=== Análisis de homocedasticidad para modelo OU ===")
            self.es_homocedastico_OU = self.auxiliar_paso5('OU')
        else:
            print("\nAdvertencia: No se ha determinado el tipo de modelo en el paso 2")

    ####################################
    # Paso 6: Estimación de parámetros #
    ####################################
    def auxiliar_paso6(self, gamma = 0):
        Xdif1 = self.X[1:]
        Xnodfi1 = self.X[:-1]

        A = np.sum(Xdif1*Xnodfi1 / Xnodfi1**(2*gamma))
        B = np.sum(Xnodfi1 / Xnodfi1**(2*gamma))
        C = np.sum(Xdif1 / Xnodfi1**(2*gamma))
        D = np.sum(1 / Xnodfi1**(2*gamma))
        E = np.sum((Xnodfi1 / Xnodfi1**gamma)**2)

        alfa_est = (E*D-B**2-A*D+B*C)/((E*D-B**2)*self.dt)
        mu_est = (A-E*(1-alfa_est*self.dt))/(alfa_est*B*self.dt)
        suma = np.sum(((Xdif1 - Xnodfi1 - alfa_est*(mu_est - Xnodfi1)*self.dt) / Xnodfi1**gamma)**2)
        sigma_est = np.sqrt((1/((len(self.X)-1)*self.dt)) * suma)

        return alfa_est, mu_est, sigma_est

    def paso6(self):
        """
        Estima los parámetros de los modelos según los resultados de los pasos anteriores.

        El método determina automáticamente qué modelo(s) estimar basándose en:
        1. Si es_tendencia es True: solo estima EDELH
        2. Si es_reg_media es True: estima OU y EDEL

        Además, verifica que los modelos a estimar hayan pasado:
        - Test de normalidad (paso 4)
        - Test de homocedasticidad (paso 5)

        Attributes
        ----------
        parametros : dict
            Diccionario con los parámetros estimados para cada modelo válido.
            Las claves son 'EDELH', 'OU' y/o 'EDEL'.
            Cada modelo contiene sus parámetros específicos:
            - EDELH: {'mu': float, 'sigma': float}
            - OU/EDEL: {'alpha': float, 'mu': float, 'sigma': float}
        """
        # Verificar que se hayan ejecutado los pasos anteriores
        if not hasattr(self, 'es_tendencia') or not hasattr(self, 'es_reg_media'):
            raise ValueError("Debe ejecutar primero el paso 2 para determinar el tipo de modelo")

        if not hasattr(self, 'es_normal_EDELH') and not hasattr(self, 'es_normal_EDEL'):
            raise ValueError("Debe ejecutar primero el paso 4 para verificar normalidad")

        if not hasattr(self, 'es_homocedastico_EDELH') and not hasattr(self, 'es_homocedastico_EDEL'):
            raise ValueError("Debe ejecutar primero el paso 5 para verificar homocedasticidad")

        parametros = {}

        # Caso 1: Modelo con tendencia (EDELH)
        if self.es_tendencia:
            # Estimar parámetros de EDELH
            self.mu_est = np.mean(self.RtiH)/self.dt
            self.sigma_est = np.sqrt(np.var(self.RtiH) / self.dt)

            parametros['EDELH'] = {
                'mu': self.mu_est,
                'sigma': self.sigma_est
            }

        # Caso 2: Modelo con regresión a la media (OU y EDEL)
        elif self.es_reg_media:
            # Modelo OU (gamma = 0)
            if self.es_normal_OU and self.es_homocedastico_OU:
                gamma = 0
                alfa_est_OU, mu_est_OU, sigma_est_OU = self.auxiliar_paso6(gamma)

                parametros['OU'] = {
                    'alpha': alfa_est_OU,
                    'mu': mu_est_OU,
                    'sigma': sigma_est_OU
                }

            if self.es_normal_EDEL and self.es_homocedastico_EDEL:
                gamma = 1
                alfa_est_EDEL, mu_est_EDEL, sigma_est_EDEL = self.auxiliar_paso6(gamma)

                parametros['EDEL'] = {
                    'alpha': alfa_est_EDEL,
                    'mu': mu_est_EDEL,
                    'sigma': sigma_est_EDEL
                }

        else:
            raise ValueError("No se pudo determinar el tipo de modelo en el paso 2")

        self.parametros = parametros

    def EDELH(self, mu, sigma, B):
        return self.X[:-1] + mu*self.X[:-1]*self.dt + sigma*B[:,1:]

    def EDEL(self, alfa, mu, sigma, gamma, B):
        return self.X[:-1] + alfa*(mu - self.X[:-1])*self.dt + sigma * (self.X[:-1]**gamma) * B[:,1:]

    def simular_browniano_estandar(self, k):
        '''
            Este método simula un movimiento browniano estándar (MBE) k-dimensional
            con vector de condiciones iniciales B0 = 0 y tamaño N igual a la base de datos

            Parametros
            ----------
            k : int
                Cantidad de MBE a simular

            Returns
            -------
            B : ndarray
                Arreglo bidimensional k x N de MBE.
        '''
        B = np.zeros(shape=(k, len(self.X)))
        B[:,1:] = np.sqrt(self.dt) * np.cumsum(np.random.randn(k, len(self.X)-1), 1)
        B = np.array(B)
        return B

    #######################################
    # Paso 7: Prónostico sobre la muestra #
    #######################################
    def paso7(self, k):
        """
        Realiza el pronóstico sobre la muestra utilizando el modelo seleccionado en los pasos anteriores.

        El método determina automáticamente qué modelo usar basándose en:
        1. Si es_tendencia es True: usa EDELH
        2. Si es_reg_media es True: estima tanto OU como EDEL y compara sus MAPEs

        Parameters
        ----------
        k : int
            Número de simulaciones a realizar para el pronóstico.

        Returns
        -------
        tuple or ndarray
            Si es_tendencia es True:
                ndarray de forma (k, N) con las simulaciones de EDELH
            Si es_reg_media es True:
                tuple con (X_P_OU, X_P_EDEL) donde cada elemento es un ndarray de forma (k, N)
                con las simulaciones de cada modelo
        """
        # Verificar que se hayan ejecutado los pasos anteriores
        if not hasattr(self, 'es_tendencia') or not hasattr(self, 'es_reg_media'):
            raise ValueError("Debe ejecutar primero el paso 2 para determinar el tipo de modelo")

        if not hasattr(self, 'parametros'):
            raise ValueError("Debe ejecutar primero el paso 6 para estimar los parámetros")

        # Simular el movimiento browniano estándar
        B = self.simular_browniano_estandar(k)
        X_P = np.zeros((k, len(self.X)))
        X_P[:,0] = self.X[0]

        # Caso 1: Modelo con tendencia (EDELH)
        if self.es_tendencia:
            if 'EDELH' not in self.parametros:
                raise ValueError("No se encontraron parámetros para el modelo EDELH")

            mu = self.parametros['EDELH']['mu']
            sigma = self.parametros['EDELH']['sigma']
            X_P[:,1:] = self.EDELH(mu, sigma, B)
            return X_P

        # Caso 2: Modelo con regresión a la media
        elif self.es_reg_media:
            if 'OU' not in self.parametros and 'EDEL' not in self.parametros:
                raise ValueError("No se encontraron parámetros para ningún modelo de regresión a la media")

            # Simular modelo OU si está disponible
            if 'OU' in self.parametros:
                alfa = self.parametros['OU']['alpha']
                mu = self.parametros['OU']['mu']
                sigma = self.parametros['OU']['sigma']
                X_P_OU = X_P.copy()
                X_P_OU[:,1:] = self.EDEL(alfa, mu, sigma, 0, B)
            else:
                X_P_OU = None

            # Simular modelo EDEL si está disponible
            if 'EDEL' in self.parametros:
                alfa = self.parametros['EDEL']['alpha']
                mu = self.parametros['EDEL']['mu']
                sigma = self.parametros['EDEL']['sigma']
                X_P_EDEL = X_P.copy()
                X_P_EDEL[:,1:] = self.EDEL(alfa, mu, sigma, 1, B)
            else:
                X_P_EDEL = None

            return X_P_OU, X_P_EDEL

        else:
            raise ValueError("No se pudo determinar el tipo de modelo en el paso 2")

    ########################################
    # Paso Extra: Análisis del pronóstico #
    ########################################
    @staticmethod
    def MAPE(y_true, y_pred):
        """
        Calcula el Error Porcentual Absoluto Medio (MAPE).

        Parameters
        ----------
        y_true : ndarray
            Valores reales
        y_pred : ndarray
            Valores predichos

        Returns
        -------
        float
            Valor del MAPE
        """
        return np.mean(np.abs((y_true - y_pred) / y_true))

    def analizar_pronostico(self, X_P):
        """
        Analiza el pronóstico realizado, mostrando las simulaciones y calculando estadísticas de error.

        Parameters
        ----------
        X_P : ndarray
            Array de forma (k, N) con las simulaciones del modelo.
            k es el número de simulaciones y N es el número de puntos.

        Returns
        -------
        tuple
            (dict, matplotlib.figure.Figure)
            - dict: Estadísticas del MAPE:
                - 'min': MAPE mínimo
                - 'mean': MAPE promedio
                - 'median': MAPE mediana
                - 'max': MAPE máximo
            - Figure: Figura con el análisis del pronóstico
        """

        # Calcular MAPE para cada simulación
        MAPE_modelo = np.array([self.MAPE(self.X, X_est) for X_est in X_P])*100

        # Calcular estadísticas del MAPE
        estadisticos_MAPE = {
            'min': np.min(MAPE_modelo),
            'mean': np.mean(MAPE_modelo),
            'median': np.median(MAPE_modelo),
            'max': np.max(MAPE_modelo)
        }

        # Crear figura con dos subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Subplot 1: Simulaciones y datos originales
        tiempo = np.arange(len(self.X))
        ax1.plot(tiempo, self.X, 'k-', label='Datos originales', linewidth=2)

        # Graficar todas las simulaciones con transparencia
        ax1.plot(tiempo, X_P.T, 'b-', alpha=0.01, linewidth=0.75)

        # Graficar la simulación con mejor MAPE
        mejor_sim = X_P[np.argmin(MAPE_modelo)]
        ax1.plot(tiempo, mejor_sim, 'r--', label='Mejor simulación', linewidth=1)

        ax1.set_xlabel(f'Tiempo en {self.frecuencia}')
        ax1.set_ylabel('Valor')
        ax1.set_title(f'Pronóstico - {self.name}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim(np.min(tiempo), np.max(tiempo))

        # Subplot 2: Histograma del MAPE
        ax2.hist(MAPE_modelo, bins=30, density=True, alpha=0.7)
        ax2.axvline(estadisticos_MAPE['mean'], color='r', linestyle='--',
                   label=f'Media: {estadisticos_MAPE["mean"]:.2f}%')
        ax2.axvline(estadisticos_MAPE['median'], color='g', linestyle='--',
                   label=f'Mediana: {estadisticos_MAPE["median"]:.2f}%')

        ax2.set_xlabel('MAPE (%)')
        ax2.set_ylabel('Densidad')
        ax2.set_title('Distribución del Error Porcentual Absoluto Medio (MAPE)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        self.guardar_figura(fig)

        # Imprimir estadísticas
        print("\n=== Análisis del Error (MAPE) ===")
        print(f"MAPE mínimo: {estadisticos_MAPE['min']:.5f}%")
        print(f"MAPE promedio: {estadisticos_MAPE['mean']:.5f}%")
        print(f"MAPE mediana: {estadisticos_MAPE['median']:.5f}%")
        print(f"MAPE máximo: {estadisticos_MAPE['max']:.5f}%")

        return estadisticos_MAPE, fig