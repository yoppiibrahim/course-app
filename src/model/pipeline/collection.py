import pandas as pd

from loguru import logger
from sqlalchemy import select

from config import engine
from db.db_model import RentApartments


def load_data_from_db():
    logger.info('Extracting csv file from db')
    query = select(RentApartments)
    return pd.read_sql(
        query,
        engine,
    )
