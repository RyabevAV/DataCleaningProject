import pandas as pd
import numpy as np

def find_outliers_iqr(data, feature,left=1.5,right=1.5,log_scale=False):
    """
    Находит выбросы в данных, используя метод межквартильного размаха.
    Классический метод модернизирован путём добавления:
    * возможности логарифмирования распределения
    * ручного управления количеством межквартильных размахов в обе стороны распределения
    Args:
        data (pandas.DataFrame): набор данных
        feature (str): имя признака, на основе которого происходит поиск выбросов
        left (float, optional): Количество межквартильных размахов в левую сторону распределения. По умолчанию 1,5
        right (float, optional): Количество межквартильных размахов в правую сторону распределения. По умолчанию 1,5
        log_scale (bool, optional): режим логарифмирования. По умолчанию False - логарифмирование не применяется
    Returns:
        outliers (pandas.DataFame): наблюдения, попавшие в разряд выбросов
        cleaned (pandas.DataFame): очищенные данные, из которых исключены выбросы 
    """
    if log_scale:
        if data[feature].min()<1: # - если в данных присутствуют значения меньше единицы:
            print('find_outliers_iqr: значения меньше 1 заменены на 1 для получения неотрицательных логарифмов')
            new_data=data.copy()
            new_data.loc[new_data[feature]<1, feature]=1
        x = np.log(new_data[feature])
    else:
        x = data[feature]
    quartile_1, quartile_3 = x.quantile(0.25), x.quantile(0.75),
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * left)
    upper_bound = quartile_3 + (iqr * right)
    outliers = data[(x<lower_bound) | (x > upper_bound)]
    cleaned = data[(x>lower_bound) & (x < upper_bound)]
    return outliers, cleaned

def find_outliers_z_score(data, feature, left=3, right=3, log_scale=False):
    """
    Находит выбросы в данных, используя "правило трёх сигм", по которому случайная величина не должна отличаться от среднего значения более чем на 3 СКО.
    СКО - среднее квадратическое отклонение (оно же "сигма") 
    Классический метод модернизирован путём добавления:
    * возможности логарифмирования распределения
    * ручного управления количеством СКО в обе стороны распределения
    Args:
        data (pandas.DataFrame): набор данных
        feature (str): имя признака, на основе которого происходит поиск выбросов
        left (float, optional): Количество СКО в левую сторону распределения. По умолчанию 3
        right (float, optional): Количество СКО в правую сторону распределения. По умолчанию 3
        log_scale (bool, optional): режим логарифмирования. По умолчанию False - логарифмирование не применяется
    Returns:
        outliers (pandas.DataFame): наблюдения, попавшие в разряд выбросов
        cleaned (pandas.DataFame): очищенные данные, из которых исключены выбросы 
    """    
    if log_scale:
        if data[feature].min()<1: # - если в данных присутствуют значения меньше единицы:
            print('find_outliers_z_score: значения меньше 1 заменены на 1 для получения неотрицательных логарифмов')
            new_data=data.copy()
            new_data.loc[new_data[feature]<1, feature]=1
        x = np.log(new_data[feature])
    else:
        x = data[feature]
    mu = x.mean()
    sigma = x.std()
    lower_bound = mu - left * sigma
    upper_bound = mu + right * sigma
    outliers = data[(x < lower_bound) | (x > upper_bound)]
    cleaned = data[(x > lower_bound) & (x < upper_bound)]
    return outliers, cleaned

def new_function():
    pass