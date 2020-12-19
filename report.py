#!/usr/bin/python3
""" xdrip+ report generator for generating printable reports out of xdrip+ database files.
    Early alpha version, only tested with a very limited set of input data from G6 sensors
"""

import argparse
import math
import sqlite3
import datetime

from typing import List, Tuple
from statistics import stdev

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

import magic

# measurement period in minutes
_PERIOD = 5

# BG plot y range
_MINLEVEL = 40
_MAXLEVEL = 225

# colorize high/low range
_LOWLEVEL = 70
_HIGHLEVEL = 170

# ignore any readings from the first 10h after sensor start
_NEW_SENSOR_INVALIDATION_PERIOD = 3600*10

# used for insulin vs. carbs ratio
_GRAMS_PER_UNIT = 10

_BOTTOMLINE = "xdrip+ report generator v0.2 Dec 2020 Andreas Fiessler/gfornax"

# lower boundary, used with 100-X for corresponding upper boundary
_PERCENTILE_1 = 5
_PERCENTILE_2 = 25

# Default charts per page
_DEFAULT_HORSIZE = 2
_DEFAULT_VERSIZE = 3

# SQL query parameters
_DB_FIELD_NAME_BG = 'calculated_value'
_DB_TABLE_NAME_BG = 'BgReadings'
_DB_TIMESTAMP_NAME_BG = 'timestamp'
_DB_FIELD_NAME_SENSOR_START = '_id'
_DB_TABLE_NAME_SENSORS = 'Sensors'
_DB_STARTED_NAME_SENSORS = 'started_at'
_DB_FIELD_NAME_TREATMENTS = 'carbs, insulin'
_DB_TABLE_NAME_TREATMENTS = 'Treatments'
_DB_TIMESTAMP_NAME_TREATMENTS = 'timestamp'

class SlotData:
    """ container objects for storing treatments, readings and events
        for time slots
    """
    def __init__(self,
                 timestamp: datetime.datetime,
                 bgval: float = -1,
                 newsensor: bool = False,
                 bolus: float = 0.0,
                 carbs: float = 0.0,
                 iob: float = 0.0,
                 invalidated: bool = False):
        self.bgval: float = bgval
        self.timestamp: datetime.datetime = timestamp
        self.newsensor: bool = newsensor
        self.bolus: float = bolus
        self.carbs: float = carbs
        self.iob: float = iob
        self.invalidated: float = invalidated
    def bgval_mmol(self) -> float:
        """ returns current value as mmol
        """
        return self.bgval * 0.0555

class DayReadings:
    """ stores the readings of exactly one day, divided in equally sized slots.
        Provides statistics and plotting methods
    """
    def __init__(self, time: datetime.datetime, slots: int):
        self.dayvalues: List[SlotData] = []
        self.startdate = time
        self.timedelta = 24*60//slots # in minutes
        self.validarray: List[float] = []
        datecursor = self.startdate
        for _ in range(slots):
            slotreading = SlotData(timestamp=datecursor)
            datecursor += datetime.timedelta(minutes=self.timedelta)
            self.dayvalues.append(slotreading)

    def update_statistics(self) -> None:
        """ parses all values, discards invalid readings (-1)
        """
        self.validarray = []
        for reading in self.dayvalues:
            if reading.bgval > 0 and not reading.invalidated:
                self.validarray.append(reading.bgval)

    def avg(self) -> float:
        """ returns average of day, ignoring invalid readings
        """
        self.update_statistics()
        if len(self.validarray) == 0:
            return 0
        return sum(self.validarray)/len(self.validarray)

    def stdev(self) -> float:
        """ returns stdev of day, ignoring invalid readings
        """
        self.update_statistics()
        if len(self.validarray) == 0:
            return 0
        return stdev(self.validarray)

    def hba1c(self) -> float:
        """ returns estimated hba1c of day, ignoring invalid readings
        """
        self.update_statistics()
        localavg = self.avg()
        return (46.7 + localavg) / 28.7
        #mmol  A1c = (2.59 + average_blood_glucose) / 1.59

    def timeinrange(self) -> Tuple[float, float, float]:
        """ returns percentage of time spent in low/ok/high range
        """
        self.update_statistics()
        readings_low = 0
        readings_ok = 0
        readings_high = 0
        for reading in self.validarray:
            if reading > 0:
                if reading < _LOWLEVEL:
                    readings_low += 1
                elif reading < _HIGHLEVEL:
                    readings_ok += 1
                else:
                    readings_high += 1
        sumreadings = readings_low + readings_ok + readings_high
        if sumreadings == 0:
            return 0, 0, 0
        return readings_low/sumreadings, readings_ok/sumreadings, readings_high/sumreadings

    def total_carbs(self) -> float:
        """ returns total carbs applied on day
        """
        carbsum = 0.0
        for reading in self.dayvalues:
            carbsum += reading.carbs
        return carbsum

    def total_bolus(self) -> float:
        """ returns total bolus applied on day
        """
        insulinsum = 0.0
        for reading in self.dayvalues:
            insulinsum += reading.bolus
        return insulinsum

    def total_ratio(self) -> float:
        """ returns ratio for day
        """
        if self.total_carbs() == 0:
            return 0
        return self.total_bolus()*_GRAMS_PER_UNIT/self.total_carbs()


    def dayplot(self, ax) -> datetime.datetime:
        """ Creates BG plot for one single day plus statistical values
        """
        xaxis = []
        yaxis = []
        invalid = []
        invalidated_limits = []
        isininvalid = False
        index = 0
        for reading in self.dayvalues:
            xaxis.append(f"{reading.timestamp.hour:02d}:{reading.timestamp.minute:02d}")
            yaxis.append(reading.bgval)
            if reading.invalidated and not reading.newsensor:
                if not isininvalid:
                    invalid.append(index)
                    isininvalid = True
            else:
                if isininvalid:
                    invalid.append(index)
                    isininvalid = False
                    invalidated_limits.append(invalid)
                    invalid = []
            index += 1

        if isininvalid:
            invalid.append(index)
            invalidated_limits.append(invalid)


        firstday = self.dayvalues[0].timestamp

        ax.axhspan(_MINLEVEL, _LOWLEVEL, alpha=0.2, color='red')
        ax.axhspan(_HIGHLEVEL, _MAXLEVEL, alpha=0.2, color='yellow')
        ax.hlines(_LOWLEVEL, 0, len(xaxis), colors='k', linestyles='solid', label='low')
        ax.hlines(_HIGHLEVEL, 0, len(xaxis), colors='k', linestyles='solid', label='high')

        # leave out missed (zero) readings
        y_values = np.ma.array(yaxis)
        y_values_masked = np.ma.masked_where(y_values <= 0 , y_values)

        ax.plot(xaxis, y_values_masked, 'o', markersize=1)
        ax.margins(x=0, y=0)
        ax.axes.set_ylim(bottom=_MINLEVEL, top=_MAXLEVEL)
        ax.xaxis.set_ticks(np.arange(0, len(xaxis), 48))
        ax.set(xlabel='time of day',
               title=f"{firstday.year}-{firstday.month:02d}-{firstday.day:02d}",
               ylabel='mg/dL')
        timelow, timeok, timehigh = self.timeinrange()
        daystats =  f"AVG: {self.avg():3.2f} mg/dL STDEV: {self.stdev():3.2f} mg/dL\n"
        daystats += f"Est. HbA1c      : {self.hba1c():3.2f}%\n"
        daystats += (f"Time low/in/high: "
                     f"{timelow*100:3.2f}%/{timeok*100:3.2f}%/{timehigh*100:3.2f}%\n")
        daystats += (f"Carbs: {self.total_carbs():3.0f}g  "
                     f"Insulin: {self.total_bolus():3.1f} Ratio: {self.total_ratio():2.1f}")
        ax.text(0, -0.43, daystats,
                horizontalalignment='left',
                verticalalignment='center',
                transform=ax.transAxes,
                family='monospace')
        ax.grid()
        for invalidarea in invalidated_limits:
            ax.axvspan(invalidarea[0], invalidarea[1], alpha=0.7, color='grey')
        return firstday

class ReportReadings:
    """ provides a list of BG readings aligned to a constant period (i.e., in "slots").
        Values are parsed out of a provided database file using start
        and end delimiters. If there were multiple readings in the same
        slot, it depends on the type whether others are discarded or accumulated.
        timestamps are expected in standard UNIX format in seconds.
    """
    def __init__(self, start_time: int, end_time: int, reading_period: int):
        assert reading_period > 0, "invalid period"
        assert start_time > 0, "invalid start_time"
        assert end_time > 0, "invalid start_time"
        assert start_time + reading_period < end_time, "invalid time frame"
        self.start_time = start_time
        self.end_time = end_time

        self.start_dtime = datetime.datetime.fromtimestamp(start_time)
        self.end_dtime = datetime.datetime.fromtimestamp(end_time)
        # sanitize the beginning of the first day and end of last day just in case
        self.start_dtime_zero = self.start_dtime.replace(minute=0, second=0, hour=0)
        self.end_dtime_max = self.end_dtime.replace(minute=59, second=59, hour=23)

        self.reading_period = reading_period
        self.report_values: List[SlotData] = []
        self.report_days = (self.end_dtime_max-self.start_dtime_zero).days + 1
        self.daily_readings: List[DayReadings] = []
        self.all_readings: List[float] = []
        timecursor = self.start_dtime_zero
        for _ in range(self.report_days):
            daily_reading = DayReadings(timecursor, (24*60)//(reading_period//60))
            self.daily_readings.append(daily_reading)
            timecursor += datetime.timedelta(days=1)
            # TODO: handling of DST. Unclear how a reasonable approach looks like for BG readings.

    def insert_readings(self, dbfile):
        """ opens dbfile using sqlite3 and parses desired values by direct SQL statements.
            Fills internal data structures.
        """
        print("Parsing batabase...")
        sensor_invalidation_start = 0
        sensor_invalidation_end = 0
        xdripdb = sqlite3.connect(dbfile)
        c = xdripdb.cursor()
        # iterate through all slots in report range, fill with matching data from DB
        for daily_reading in self.daily_readings:
            for slotdata in daily_reading.dayvalues:
                match_period_start = slotdata.timestamp.timestamp()*1000
                match_period_end = (slotdata.timestamp.timestamp()+self.reading_period)*1000
                for row in c.execute(f"SELECT {_DB_FIELD_NAME_BG} "
                                     f"FROM {_DB_TABLE_NAME_BG} "
                                     f"WHERE {_DB_TIMESTAMP_NAME_BG} >= {match_period_start} "
                                     f"and {_DB_TIMESTAMP_NAME_BG} < {match_period_end}"):
                    slotdata.bgval = row[0]
                    if (match_period_start >= sensor_invalidation_start and
                        match_period_start <= sensor_invalidation_end):
                        slotdata.invalidated = True
                    else:
                        self.all_readings.append(row[0]) # collect all valid readings for overall stats
                    break
                for row in c.execute(f"SELECT {_DB_FIELD_NAME_SENSOR_START} "
                                     f"FROM {_DB_TABLE_NAME_SENSORS} "
                                     f"WHERE {_DB_STARTED_NAME_SENSORS} >= {match_period_start} "
                                     f"and {_DB_STARTED_NAME_SENSORS} < {match_period_end}"):
                    slotdata.newsensor = True
                    slotdata.invalidated = True
                    sensor_invalidation_start = match_period_start
                    sensor_invalidation_end = match_period_start + _NEW_SENSOR_INVALIDATION_PERIOD*1000
                    break
                bolus = 0
                carbs = 0
                for row in c.execute(f"SELECT {_DB_FIELD_NAME_TREATMENTS} "
                                     f"FROM {_DB_TABLE_NAME_TREATMENTS} "
                                     f"WHERE {_DB_TIMESTAMP_NAME_TREATMENTS} >= {match_period_start} "
                                     f"and {_DB_TIMESTAMP_NAME_TREATMENTS} < {match_period_end}"):

                    carbs += row[0]
                    bolus += row[1]
                slotdata.carbs = carbs
                slotdata.bolus = bolus
                # TODO: add IOB here

        xdripdb.close()

    def create_daily_stats(self, ax):
        """ create daily patterns
        """
        # arrange data for computation of patterns:
        dailyreadings = []
        for daily_reading in self.daily_readings:
            dayreadings = []
            # use zero values for invalidated slots as they can be masked easily
            for slotdata in daily_reading.dayvalues:
                if slotdata.invalidated:
                    dayreadings.append(0)
                else:
                    dayreadings.append(slotdata.bgval)
            ret_array  = np.ma.array(dayreadings)
            dayreadings_masked = np.ma.masked_where(ret_array <= 0 , ret_array)
            dailyreadings.append(dayreadings_masked)
        arranged = np.array(dailyreadings).transpose()
        daypattern_mean = []
        daypattern_median = []
        daypattern_max = []
        daypattern_1 = []
        daypattern_2 = []
        # remove invalid/missing readings
        for larr in arranged:
            validarray = np.array(list(filter(lambda a: a > 1, larr)))
            if len(validarray) == 0:
                validarray = [0]
                print("Warning: no valid readings for daily stat slot")
            daypattern_mean.append(np.mean(validarray))
            daypattern_median.append(np.percentile(validarray, 50))
            daypattern_max.append([min(validarray), max(validarray)])
            ## confidence interval:
#            daypattern_95.append(st.t.interval(0.95, len(validarray)-1,
#                                 loc=np.mean(validarray), scale=st.sem(validarray)))
#            daypattern_75.append(st.t.interval(0.75, len(validarray)-1,
#                                 loc=np.mean(validarray), scale=st.sem(validarray)))
            # percentiles
            daypattern_1.append([np.percentile(validarray, _PERCENTILE_1),
                                  np.percentile(validarray, (100-_PERCENTILE_1))])
            daypattern_2.append([np.percentile(validarray, _PERCENTILE_2),
                                  np.percentile(validarray, (100-_PERCENTILE_2))])

        xaxis = []
        for reading in self.daily_readings[0].dayvalues:
            xaxis.append(f"{reading.timestamp.hour:02d}:{reading.timestamp.minute:02d}")

        ax.axhspan(_MINLEVEL, _LOWLEVEL, alpha=0.2, color='red')
        ax.axhspan(_HIGHLEVEL, _MAXLEVEL, alpha=0.2, color='yellow')
        ax.plot(xaxis, daypattern_median, label='median')
        ax.plot(xaxis, daypattern_mean, label='mean')
        ax.fill_between(xaxis,
                        np.array(daypattern_1).transpose()[0],
                        np.array(daypattern_1).transpose()[1],
                        color='red',
                        alpha=0.99,
                        label=f"percentile {_PERCENTILE_1/100}-{(100-_PERCENTILE_1)/100}")
        ax.fill_between(xaxis,
                        np.array(daypattern_2).transpose()[0],
                        np.array(daypattern_2).transpose()[1],
                        color='blue',
                        alpha=0.99,
                        label=f"percentile {_PERCENTILE_2/100}-{(100-_PERCENTILE_2)/100}")
        ax.fill_between(xaxis,
                        np.array(daypattern_max).transpose()[0],
                        np.array(daypattern_max).transpose()[1],
                        color='green',
                        alpha=0.2, label='whiskers')
        ax.hlines(_LOWLEVEL, 0, len(xaxis), colors='k', linestyles='solid')
        ax.hlines(_HIGHLEVEL, 0, len(xaxis), colors='k', linestyles='solid')
        ax.margins(x=0, y=0)
        ax.axes.set_ylim(bottom=_MINLEVEL, top=_MAXLEVEL)
        ax.xaxis.set_ticks(np.arange(0, len(xaxis), 48))
        ax.set(xlabel='time of day', title=f"Daily Pattern", ylabel='mg/dL')
        timelow, timeok, timehigh = self.timeinrange()
        daystats  = f"AVG: {self.avg():3.2f} mg/dL STDEV: {self.stdev():3.2f} mg/dL\n"
        daystats += f"Est. HbA1c      : {self.hba1c():3.2f}%\n"
        daystats += (f"Time low/in/high: "
                     f"{timelow*100:3.2f}%/{timeok*100:3.2f}%/{timehigh*100:3.2f}%")
        ax.text(0, -0.34,
                daystats,
                horizontalalignment='left',
                verticalalignment='center',
                transform=ax.transAxes,
                family='monospace')
        ax.legend(loc='upper left')
        ax.grid()

    def create_treatment_stats(self, ax):
        """ create treatments plot
        """
        xaxis = []
        total_bolus = []
        total_carbs = []
        for daily_reading in self.daily_readings:
            total_bolus.append(daily_reading.total_bolus())
            total_carbs.append(daily_reading.total_carbs())
            xaxis.append(f"{daily_reading.startdate.year}-{daily_reading.startdate.month:02d}-"
                         f"{daily_reading.startdate.day:02d}")

        color = 'tab:red'
        ax.set_xlabel('date')
        ax.set_ylabel('Bolus per day', color=color)
        ax.plot(xaxis, total_bolus, color=color)
        ax.tick_params(axis='y', labelcolor=color)

        ax2 = ax.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('carbs [g]', color=color)
        ax2.plot(xaxis, total_carbs, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        ax.margins(x=0, y=0)

        ax.xaxis.set_ticks(np.arange(0, len(xaxis), len(xaxis)/5))
        ax.set(xlabel='date', title=f"Carbs and bolus by day", ylabel='bolus')
        daystats  = (f"Carbs Min/AVG/Max: "
                     f"{min(total_carbs):3.0f}g/{sum(total_carbs)/len(total_carbs):3.0f}g"
                     f"/{max(total_carbs):3.0f}g\n")
        daystats += (f"Bolus Min/AVG/Max: {min(total_bolus):3.0f} "
                     f"/{sum(total_bolus)/len(total_bolus):3.0f} /{max(total_bolus):3.0f}\n")
        ax.text(0, -0.34,
                daystats,
                horizontalalignment='left',
                verticalalignment='center',
                transform=ax.transAxes,
                family='monospace')
        ax.grid()

    def create_ratio_stats(self, ax):
        """ create ratio plot
        """
        xaxis = []
        total_ratio = []
        for daily_reading in self.daily_readings:
            total_ratio.append(daily_reading.total_ratio())
            xaxis.append(f"{daily_reading.startdate.year}-{daily_reading.startdate.month:02d}-"
                         f"{daily_reading.startdate.day:02d}")

        color = 'tab:red'
        ax.set_xlabel('date')
        ax.set_ylabel('Ratio per day', color=color)
        ax.plot(xaxis, total_ratio, color=color)
        ax.tick_params(axis='y', labelcolor=color)

        ax.margins(x=0, y=0)

        ax.xaxis.set_ticks(np.arange(0, len(xaxis), len(xaxis)/5))
        ax.set(xlabel='date', title=f"Ratio", ylabel=f"Units/{_GRAMS_PER_UNIT}g carbs")
        daystats = (f"Ratio Min/AVG/Max: {min(total_ratio):1.1f} "
                    f"/{sum(total_ratio)/len(total_ratio):1.1f} /{max(total_ratio):1.1f}\n")
        ax.text(0, -0.34,
                daystats,
                horizontalalignment='left',
                verticalalignment='center',
                transform=ax.transAxes,
                family='monospace')
        ax.grid()

    def avg(self) -> float:
        """ returns total average
        """
        validarray = list(filter(lambda a: a != -1, self.all_readings))
        return sum(validarray)/len(validarray)

    def stdev(self) -> float:
        """ returns stddev, all report days
        """
        validarray = list(filter(lambda a: a != -1, self.all_readings))
        return stdev(validarray)

    def hba1c(self) -> float:
        """ returns estimated hba1c, all report days
        """
        localavg = self.avg()
        return (46.7 + localavg) / 28.7

    def timeinrange(self):
        """ time in range statistics, all report days
        """
        readings_low = 0
        readings_ok = 0
        readings_high = 0
        validarray = list(filter(lambda a: a != -1, self.all_readings))
        for reading in validarray:
            if reading < _LOWLEVEL:
                readings_low += 1
            elif reading < _HIGHLEVEL:
                readings_ok += 1
            else:
                readings_high += 1
        sumreadings = readings_low + readings_ok + readings_high
        return readings_low/sumreadings, readings_ok/sumreadings, readings_high/sumreadings

    def create_report_page(self, patname: str, filename: str, rows: int, columns: int):
        """ create a report pdf by passing the subfigure objects to corresponding objects
            plot methods
        """
        #calculate number of pages
        PLOTS_PER_PAGE = rows * columns
        numpages = math.ceil(self.report_days/PLOTS_PER_PAGE)
        print(f"report days: {self.report_days}, numpages: {numpages}")
        dayindex = 0 ## use enum index for future day selection in report
        with PdfPages(filename) as pdf:
            fig, ax = plt.subplots(3)
            fig.subplots_adjust(bottom=0.15, top=0.93, right=0.90, hspace=0.62)
            fig.set_size_inches(w=8.2, h=11.6)
            self.create_daily_stats(ax.flat[0])
            self.create_treatment_stats(ax.flat[1])
            self.create_ratio_stats(ax.flat[2])
            fig.suptitle(f"Statistics {self.start_dtime.year}-{self.start_dtime.month:02d}-"
                         f"{self.start_dtime.day:02d} to {self.end_dtime.year}-"
                         f"{self.end_dtime.month:02d}-{self.end_dtime.day:02d}    "
                         f"Pat. {patname}", fontsize=12)
            fig.text(.1, 0.03, _BOTTOMLINE, fontsize=8)
            pdf.savefig()
            plt.close()

            for page in range(numpages):
                daysinplot = []
                fig, ax = plt.subplots(rows, columns)
                fig.subplots_adjust(bottom=0.15, top=0.93, right=0.95, hspace=0.80)
                fig.set_size_inches(w=8.2, h=11.6)
                for ax_cur in ax.flat:
                    if dayindex >= self.report_days:
                        break
                    daysinplot.append(self.daily_readings[dayindex].dayplot(ax_cur))
                    dayindex += 1
                fig.suptitle(f"BG report {daysinplot[0].year}-{daysinplot[0].month:02d}-"
                             f"{daysinplot[0].day:02d} to "
                             f"{daysinplot[-1].year}-{daysinplot[-1].month:02d}-"
                             f"{daysinplot[-1].day:02d} "
                             f"Pat. {patname}", fontsize=12)
                fig.text(.1, 0.03, _BOTTOMLINE, fontsize=8)
                fig.text(.85, 0.03, f"page {page+1}/{numpages}", fontsize=8)
                pdf.savefig()
                plt.close()

    def print_readings(self):
        """ debug function: print all parsed readings
        """
        for reading in self.report_values:
            print(f"value: {reading.bgval} on {datetime.datetime.fromtimestamp(reading.timestamp)}")

def parse_args() -> Tuple[str, str, datetime.datetime, datetime.datetime, str]:
    """ parses/sanitizes CMD line args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dbfile", help="name of sqlite database file")
    parser.add_argument("-n", "--name", help="Patient name on report")
    parser.add_argument("-s", "--start", help="Report start day YYYY-MM-DD")
    parser.add_argument("-e", "--end", help="Report end day YYYY-MM-DD")
    parser.add_argument("-f", "--filename", help="output PDF file name")
    parser.add_argument("-g", "--grid", help="Layout of chart grid. WxH = W per row, H rows per page")
    args = parser.parse_args()

    rows = _DEFAULT_VERSIZE
    columns = _DEFAULT_HORSIZE

    if args.dbfile:
        dbfile = args.dbfile
        assert magic.from_file(dbfile, mime=True) == 'application/x-sqlite3', "Unsupported DB type"
    else:
        parser.print_help()
        exit(1)
    if args.name:
        patname = args.name
    else:
        parser.print_help()
        exit(1)
    if args.filename:
        filename = args.filename
    else:
        parser.print_help()
        exit(1)
    if args.start:
        if (len(args.start) != 10):
            parser.print_help()
            exit(1)
        syear = args.start[:4]
        smonth = args.start[5:7]
        sday = args.start[8:10]
        if syear.isdigit() and smonth.isdigit() and sday.isdigit():
            stime = datetime.datetime(year=int(syear), month=int(smonth), day=int(sday))
        else:
            parser.print_help()
            exit(1)
    else:
        parser.print_help()
        exit(1)
    if args.end:
        if (len(args.end) != 10):
            parser.print_help()
            exit(1)
        eyear = args.end[:4]
        emonth = args.end[5:7]
        eday = args.end[8:10]
        if eyear.isdigit() and emonth.isdigit() and eday.isdigit():
            etime = datetime.datetime(year=int(eyear), month=int(emonth), day=int(eday))
        else:
            parser.print_help()
            exit(1)
    else:
        parser.print_help()
        exit(1)

    if args.grid:
        columns, rows = [int(n) for n in args.grid.split("x")]

    return dbfile, patname, stime, etime, filename, rows, columns

if __name__ == '__main__':
    dbfile, patname, stime, etime, filename, rows, columns = parse_args()
    report = ReportReadings(int(stime.timestamp()), int(etime.timestamp()), 60*_PERIOD)
    report.insert_readings(dbfile)
    print("Creating report PDF")
    report.create_report_page(patname, filename, rows, columns)
