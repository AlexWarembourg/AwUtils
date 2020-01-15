import calendar

import pandas as pd
from src.data_utils.utils import *
from pandas.tseries.holiday import *
from pandas.tseries.offsets import CustomBusinessDay


class UACalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('New Year', month=1, day=1),
        Holiday('Orthodox Christmas', month=1, day=7),
        Holiday('Malanka Day start', month=1, day=13, offset=[Day(-1)]),
        Holiday('Malanka Day end', month=1, day=13, offset=[Day(1)]),
        Holiday('Unity Day', month=1, day=22),
        Holiday('Tatiana Day', month=1, day=25),
        Holiday('Valentine Day', month=2, day=14),
        Holiday('Womens Day', month=3, day=8),
        Holiday('April Fools Day', month=4, day=1),
        Holiday('Good Friday', month=1, day=1, offset=[Easter(), Day(-2)]),  # easter friday
        Holiday('Labor Day', month=5, day=1),
        Holiday('Victory Day', month=5, day=9),
        Holiday('Europe Day', month=5, day=18),
        Holiday('Kiev Day', month=5, day=26),
        Holiday('Pentecost Day', month=1, day=1, offset=[Easter(), Day(50)]),  # pentecost friday
        Holiday('Constitution Day', month=6, day=28),
        Holiday('Independance Day', month=8, day=24),  # to expand to 3 day observed
        Holiday('Teachers Day3', month=10, day=6),
        Holiday('Defender Day', month=10, day=14),
        Holiday('Dignity and Freedom Day', month=11, day=21),
        Holiday('Armed Forces Day', month=12, day=6),
        Holiday('Christmas', month=12, day=25)
    ]


# mother dat each nd sunday of mai
mother_day = [pd.to_datetime(str(x) + '-05-' + str(calendar.monthcalendar(int(x), 5)[1][calendar.FRIDAY])) for x in
              range(2000, 2051, 1)]

# father day each third sunday of june
father_day = [pd.to_datetime(str(x) + '-06-' + str(calendar.monthcalendar(int(x), 6)[2][calendar.FRIDAY])) for x in
              range(2000, 2051, 1)]

if __name__ == "__main__":
    UABD = CustomBusinessDay(calendar=UACalendar())

    calendar = (pd.DataFrame(
        pd.date_range('2000-01-01', end='2050-01-01'), columns=['dt_ticket_sale'])
                .assign(dt_ticket_sale=lambda x: x["dt_ticket_sale"].astype('datetime64[ns]'))
                )

    holiday = (UACalendar()
               .holidays(start='2016-01-01', end='2025-12-31', return_name=True)
               .reset_index()
               .rename(columns={"index": "dt_ticket_sale", 0: "Holiday_Name"})
               .assign(dt_ticket_sale=lambda x: x["dt_ticket_sale"].astype('datetime64[ns]'))
               )

    calendar_ = calendar.merge(holiday, on="dt_ticket_sale", how="outer")

    calendar_['Holiday_Name'] = np.where(calendar_["dt_ticket_sale"].isin(father_day), "Father_Day",
                                         calendar_["Holiday_Name"])

    calendar_['Holiday_Name'] = np.where(calendar_["dt_ticket_sale"].isin(mother_day), "Mother_Day",
                                         calendar_["Holiday_Name"])

    improved_calendar = add_date_features(calendar_, calendar_["dt_ticket_sale"], weeks=1)

    # [TODO] : Implement time elapsed since
    # expand spiky holiday
    improved_calendar["xmas_flg"] = np.where(improved_calendar["Holiday_Name"] == "Christmas", 1, 0)
    improved_calendar["xmas_flg"] = expend_event(improved_calendar, "xmas_flg", direction="both", n_ahead=3,
                                                 n_previous=10)

    improved_calendar["easter_flg"] = np.where(improved_calendar["Holiday_Name"] == 'Good Friday', 1, 0)
    improved_calendar["easter_flg"] = expend_event(improved_calendar, "easter_flg", direction="both", n_ahead=5,
                                                   n_previous=10)

    improved_calendar["orthodox_xmas_flg"] = np.where(improved_calendar["Holiday_Name"] == 'Orthodox Christmas', 1, 0)
    improved_calendar["orthodox_xmas_flg"] = expend_event(improved_calendar, "orthodox_xmas_flg", direction="both",
                                                          n_ahead=3,
                                                          n_previous=6)

    improved_calendar.to_csv("/home/alex/TopDownForecast/data/calendar.csv", index=False)
