import os

import influxdb_client
import datetime as dt
import numpy as np
import pandas as pd
from dotenv import dotenv_values

class NoDataException(Exception):
    pass


class DataBaseConnection:
    def __init__(self, env_file=".env"):
        env_values = dotenv_values(env_file)
        self.bucket = env_values['BUCKET']
        self.org = env_values['ORG']
        self.token = env_values['INFLUXDB_TOKEN']
        self.url = env_values['URL']
        self.client = influxdb_client.InfluxDBClient(url=self.url, token=self.token, org=self.org)
        self.query_api = self.client.query_api()

    def query_database_since(self, data_since='1m', bucket='Sciteas2', field='temperature_copper_plate'):
        """Query a database field during the specified time period."""

        query = f'from(bucket: "{bucket}")\
        |> range(start: -{data_since})\
        |> filter(fn: (r) => r["_field"] == "{field}")'

        result = self.query_api.query(query=query)

        temperatures = []
        for table in result:
            for record in table.records:
                temperatures.append(record.values['_value'])

        if not temperatures:
            raise NoDataException("No data found in the specified time range.")

        return sum(temperatures) / len(temperatures), min(temperatures), max(temperatures)

    def query_database_timerange(self, start_timestamp, stop_timestamp, bucket='Sciteas2',
                                 field='temperature_copper_plate'):
        """Query a database field during the specified time period."""

        query = f'from(bucket: "{bucket}")\
        |> range(start: {int(start_timestamp)}, stop: {int(stop_timestamp)})\
        |> filter(fn: (r) => r["_field"] == "{field}")'

        result = self.query_api.query(query=query)

        temperatures = []
        for table in result:
            for record in table.records:
                temperatures.append(record.values['_value'])

        if len(temperatures) == 0:
            raise NoDataException("No data found in the specified time range.")

        return sum(temperatures) / len(temperatures), min(temperatures), max(temperatures)


if __name__ == '__main__':
    db = DataBaseConnection()

    start = int(dt.datetime.strptime("01.08.23 22:40", "%d.%m.%y %H:%M").strftime("%s"))
    end = int(dt.datetime.strptime("03.08.23 10:40", "%d.%m.%y %H:%M").strftime("%s"))
    diode_id = 1
    name = f"warmup_id={diode_id}_1"

    bucket = "Sciteas2"
    fields = ["diode_v_10ua", "diode_t_10ua", "diode_v_100ua",
              "diode_t_100ua"]

    data_diode = {}
    for field in fields:
        if "diode" in field:
            query = f'from(bucket: "{bucket}")\
                        |> range(start: {start}, stop: {end})\
                            |> filter(fn: (r) => r["_field"] == "{field}") \
                    |> filter(fn: (r) => r["diode_id"] == "{diode_id}")'
        else:
            query = f'from(bucket: "{bucket}")\
                                    |> range(start: {start}, stop: {end})\
                                        |> filter(fn: (r) => r["_field"] == "{field}")'
        result = db.query_api.query(query=query)
        dat = []
        timestamp = []
        for table in result:
            for record in table.records:
                dat.append(record.values['_value'])
                if field == fields[0]:
                    timestamp.append(record.values['_time'])
        data_diode[field] = dat
        if field == fields[0]:
            data_diode["time"] = timestamp
        print(field, len(dat))
        if not dat:
            raise NoDataException(f"No data found for {field} in the specified time range.")

    fields = ["pt1000_adc", "pt_1000_t"]
    data_pt = {}
    for field in fields:
        query = f'from(bucket: "{bucket}")\
                                       |> range(start: {start}, stop: {end})\
                                           |> filter(fn: (r) => r["_field"] == "{field}")'
        result = db.query_api.query(query=query)
        dat = []
        timestamp = []
        for table in result:
            for record in table.records:
                dat.append(record.values['_value'])
                if field == fields[0]:
                    timestamp.append(record.values['_time'])
        data_pt[field] = dat
        if field == fields[0]:
            data_pt["time"] = timestamp
        print(field, len(dat))
        if not dat:
            raise NoDataException(f"No data found for {field} in the specified time range.")

    query = f'from(bucket: "{bucket}")\
                                        |> range(start: {start}, stop: {end})\
                                            |> filter(fn: (r) => r["_field"] == "temperature_lakeshore")'
    result = db.query_api.query(query=query)
    data_lakeshore = {}
    dat = []
    timestamp = []
    for table in result:
        for record in table.records:
            dat.append(record.values['_value'])
            timestamp.append(record.values['_time'])
    data_lakeshore["time"] = timestamp
    data_lakeshore["temperature_lakeshore"] = dat

    data_pt["pt_1000_t"] = data_pt["pt_1000_t"][:len(data_pt["pt_1000_t"])]
    data_pt["pt1000_adc"] = data_pt["pt1000_adc"][:len(data_pt["pt_1000_t"])]
    data_pt["time"] = data_pt["time"][:len(data_pt["pt_1000_t"])]
    data_diode["time"] = data_diode["time"][:len(data_diode["diode_v_10ua"])]
    data_diode["diode_v_10ua"] = data_diode["diode_v_10ua"][:len(data_diode["diode_v_10ua"])]
    data_diode["diode_t_10ua"] = data_diode["diode_t_10ua"][:len(data_diode["diode_v_10ua"])]
    data_diode["diode_v_100ua"] = data_diode["diode_v_100ua"][:len(data_diode["diode_v_10ua"])]
    data_diode["diode_t_100ua"] = data_diode["diode_t_100ua"][:len(data_diode["diode_v_10ua"])]
    df_diode = pd.DataFrame(data=data_diode)
    df_pt = pd.DataFrame(data=data_pt)
    df_lakeshore = pd.DataFrame(data=data_lakeshore)
    os.mkdir(f"data/{name}")
    df_diode.to_csv(f"data/{name}/diode.csv")
    df_pt.to_csv(f"data/{name}/pt.csv")
    df_lakeshore.to_csv(f"data/{name}/lakeshore.csv")
