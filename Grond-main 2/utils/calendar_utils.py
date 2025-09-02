# utils/calendar_utils.py

from datetime import datetime, time as dtime
import pandas_market_calendars as mcal
from config import tz

nyse = mcal.get_calendar("NYSE")

def is_market_open_today(as_of: datetime | None = None) -> bool:
    """
    Return True if today is a valid trading day.
    """
    d = (as_of or datetime.now(tz)).date()
    return not nyse.valid_days(start_date=d, end_date=d).empty

def calculate_time_of_day(as_of: datetime | None = None) -> str:
    """
    Bucket current time into PRE_MARKET, MORNING, MIDDAY, AFTERNOON, AFTER_HOURS, OFF_HOURS.
    """
    now = (as_of or datetime.now(tz)).time()
    if dtime(4,0) <= now < dtime(9,30):   return "PRE_MARKET"
    if dtime(9,30) <= now < dtime(11,0):  return "MORNING"
    if dtime(11,0) <= now < dtime(14,0):  return "MIDDAY"
    if dtime(14,0) <= now < dtime(16,0):  return "AFTERNOON"
    if dtime(16,0) <= now < dtime(20,0):  return "AFTER_HOURS"
    return "OFF_HOURS"
