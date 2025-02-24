import re

import pandas as pd
from loguru import logger

from model.pipeline.collection import load_data_from_db


def prepare_data() -> pd.DataFrame:
    logger.info('Staring up preparation data Pipeline')
    dataframe = load_data_from_db()
    encode_data = _encode_cat_cols(dataframe)
    df = parse_garden_col(encode_data)
    return df


def _encode_cat_cols(dataframe) -> pd.DataFrame:
    cols = ['balcony', 'parking', 'furnished', 'garage', 'storage']
    logger.info(f'Encoding categorical columns {cols}')

    return pd.get_dummies(
        dataframe,
        columns=cols,
        dtype=int,
        drop_first=True,
    )


def parse_garden_col(dataframe) -> pd.DataFrame:
    logger.info('Parsing garden column')
    dataframe['garden'] = dataframe['garden'].apply(
        lambda x: 0 if x == 'Not present' else int(re.findall(r'/d+', x)[0])
    )
    return dataframe
