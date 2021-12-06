from numpy import nan
import pandas as pd
from pandas import Timestamp

expected_data = pd.DataFrame(
    [
        {
            "open_time": Timestamp("2021-04-21 14:00:00+0000", tz="UTC"),
            "close_time": Timestamp("2021-04-21 13:59:59.999000+0000", tz="UTC"),
            "open": 55306.46,
            "high": 55399.68,
            "low": 55217.22,
            "close": 55388.96,
            "volume": 276.690734,
            "quote_volume": 15295597.50785806,
            "trades": 7614,
            "taker_buy_asset_volume": 145.211424,
            "taker_buy_quote_volume": 8027028.99029815,
            "returns": nan,
            "sma": nan,
            "upper": nan,
            "lower": nan,
        },
        {
            "open_time": Timestamp("2021-04-21 14:05:00+0000", tz="UTC"),
            "close_time": Timestamp("2021-04-21 14:04:59.999000+0000", tz="UTC"),
            "open": 55306.46,
            "high": 55399.68,
            "low": 55217.22,
            "close": 55388.96,
            "volume": 276.690734,
            "quote_volume": 15295597.50785806,
            "trades": 7614,
            "taker_buy_asset_volume": 145.211424,
            "taker_buy_quote_volume": 8027028.99029815,
            "returns": 0.0,
            "sma": nan,
            "upper": nan,
            "lower": nan,
        },
        {
            "open_time": Timestamp("2021-04-21 14:10:00+0000", tz="UTC"),
            "close_time": Timestamp("2021-04-21 14:09:59.999000+0000", tz="UTC"),
            "open": 55306.46,
            "high": 55399.68,
            "low": 55217.22,
            "close": 55388.96,
            "volume": 276.690734,
            "quote_volume": 15295597.50785806,
            "trades": 7614,
            "taker_buy_asset_volume": 145.211424,
            "taker_buy_quote_volume": 8027028.99029815,
            "returns": 0.0,
            "sma": nan,
            "upper": nan,
            "lower": nan,
        },
        {
            "open_time": Timestamp("2021-04-21 14:15:00+0000", tz="UTC"),
            "close_time": Timestamp("2021-04-21 14:14:59.999000+0000", tz="UTC"),
            "open": 55306.46,
            "high": 55399.68,
            "low": 55217.22,
            "close": 55388.96,
            "volume": 276.690734,
            "quote_volume": 15295597.50785806,
            "trades": 7614,
            "taker_buy_asset_volume": 145.211424,
            "taker_buy_quote_volume": 8027028.99029815,
            "returns": 0.0,
            "sma": nan,
            "upper": nan,
            "lower": nan,
        },
        {
            "open_time": Timestamp("2021-04-21 14:20:00+0000", tz="UTC"),
            "close_time": Timestamp("2021-04-21 14:19:59.999000+0000", tz="UTC"),
            "open": 55306.46,
            "high": 55399.68,
            "low": 55217.22,
            "close": 55388.96,
            "volume": 276.690734,
            "quote_volume": 15295597.50785806,
            "trades": 7614,
            "taker_buy_asset_volume": 145.211424,
            "taker_buy_quote_volume": 8027028.99029815,
            "returns": 0.0,
            "sma": nan,
            "upper": nan,
            "lower": nan,
        },
        {
            "open_time": Timestamp("2021-04-21 14:25:00+0000", tz="UTC"),
            "close_time": Timestamp("2021-04-21 14:29:59.999000+0000", tz="UTC"),
            "open": 55388.95,
            "high": 55569.95,
            "low": 55388.95,
            "close": 55552.4,
            "volume": 149.363426,
            "quote_volume": 8288967.03877351,
            "trades": 5260,
            "taker_buy_asset_volume": 82.67909,
            "taker_buy_quote_volume": 4588065.23181743,
            "returns": 0.002946423556387118,
            "sma": nan,
            "upper": nan,
            "lower": nan,
        },
        {
            "open_time": Timestamp("2021-04-21 14:30:00+0000", tz="UTC"),
            "close_time": Timestamp("2021-04-21 14:34:59.999000+0000", tz="UTC"),
            "open": 55550.89,
            "high": 56087.68,
            "low": 55550.89,
            "close": 55932.48,
            "volume": 692.924319,
            "quote_volume": 38726480.58078431,
            "trades": 16507,
            "taker_buy_asset_volume": 411.223017,
            "taker_buy_quote_volume": 22979821.33981915,
            "returns": 0.006818529518377586,
            "sma": 55489.95428571429,
            "upper": 55898.796312069564,
            "lower": 55081.11225935901,
        },
        {
            "open_time": Timestamp("2021-04-21 14:35:00+0000", tz="UTC"),
            "close_time": Timestamp("2021-04-21 14:39:59.999000+0000", tz="UTC"),
            "open": 55932.48,
            "high": 56333.0,
            "low": 55932.48,
            "close": 56264.93,
            "volume": 603.660118,
            "quote_volume": 33896505.6971466,
            "trades": 14656,
            "taker_buy_asset_volume": 356.915883,
            "taker_buy_quote_volume": 20037884.70780964,
            "returns": 0.005926179097336494,
            "sma": 55615.09285714285,
            "upper": 56313.42304711646,
            "lower": 54916.76266716924,
        },
        {
            "open_time": Timestamp("2021-04-21 14:40:00+0000", tz="UTC"),
            "close_time": Timestamp("2021-04-21 14:44:59.999000+0000", tz="UTC"),
            "open": 56260.11,
            "high": 56317.43,
            "low": 56118.31,
            "close": 56168.82,
            "volume": 370.500359,
            "quote_volume": 20822485.25288953,
            "trades": 8616,
            "taker_buy_asset_volume": 178.075904,
            "taker_buy_quote_volume": 10007121.78892216,
            "returns": -0.00170962942015996,
            "sma": 55726.50142857143,
            "upper": 56501.13776151988,
            "lower": 54951.865095622976,
        },
        {
            "open_time": Timestamp("2021-04-21 14:45:00+0000", tz="UTC"),
            "close_time": Timestamp("2021-04-21 14:49:59.999000+0000", tz="UTC"),
            "open": 56168.82,
            "high": 56269.99,
            "low": 56080.96,
            "close": 56191.11,
            "volume": 324.51432,
            "quote_volume": 18225087.41727558,
            "trades": 8352,
            "taker_buy_asset_volume": 145.064381,
            "taker_buy_quote_volume": 8146012.47892439,
            "returns": 0.0003967606653441501,
            "sma": 55841.09428571428,
            "upper": 56620.02450114715,
            "lower": 55062.16407028141,
        },
        {
            "open_time": Timestamp("2021-04-21 14:50:00+0000", tz="UTC"),
            "close_time": Timestamp("2021-04-21 14:54:59.999000+0000", tz="UTC"),
            "open": 56191.11,
            "high": 56200.0,
            "low": 56107.98,
            "close": 56145.0,
            "volume": 254.091606,
            "quote_volume": 14265787.89818125,
            "trades": 6455,
            "taker_buy_asset_volume": 134.1124,
            "taker_buy_quote_volume": 7529521.30623853,
            "returns": -0.0008209293091875578,
            "sma": 55949.09999999999,
            "upper": 56640.1740680998,
            "lower": 55258.025931900185,
        },
        {
            "open_time": Timestamp("2021-04-21 14:55:00+0000", tz="UTC"),
            "close_time": Timestamp("2021-04-21 14:59:59.999000+0000", tz="UTC"),
            "open": 56145.0,
            "high": 56211.7,
            "low": 56106.97,
            "close": 56182.11,
            "volume": 270.145731,
            "quote_volume": 15171017.18758856,
            "trades": 7707,
            "taker_buy_asset_volume": 168.231118,
            "taker_buy_quote_volume": 9447425.0774598,
            "returns": 0.0006607487960858385,
            "sma": 56062.40714285715,
            "upper": 56557.0735836991,
            "lower": 55567.740702015195,
        },
        {
            "open_time": Timestamp("2021-04-21 15:00:00+0000", tz="UTC"),
            "close_time": Timestamp("2021-04-21 15:04:59.999000+0000", tz="UTC"),
            "open": 56182.12,
            "high": 56299.78,
            "low": 56172.09,
            "close": 56289.89,
            "volume": 298.797415,
            "quote_volume": 16804824.55255641,
            "trades": 9000,
            "taker_buy_asset_volume": 139.83665,
            "taker_buy_quote_volume": 7864202.02549528,
            "returns": 0.0019165664875115606,
            "sma": 56167.76285714285,
            "upper": 56400.11992481078,
            "lower": 55935.40578947492,
        },
    ]
)

expected_data_set_parameters = pd.DataFrame(
    [
        {
            "open_time": Timestamp("2021-04-21 14:00:00+0000", tz="UTC"),
            "close_time": Timestamp("2021-04-21 13:59:59.999000+0000", tz="UTC"),
            "open": 55306.46,
            "high": 55399.68,
            "low": 55217.22,
            "close": 55388.96,
            "volume": 276.690734,
            "quote_volume": 15295597.50785806,
            "trades": 7614,
            "taker_buy_asset_volume": 145.211424,
            "taker_buy_quote_volume": 8027028.99029815,
            "returns": nan,
            "sma": nan,
            "upper": nan,
            "lower": nan,
        },
        {
            "open_time": Timestamp("2021-04-21 14:05:00+0000", tz="UTC"),
            "close_time": Timestamp("2021-04-21 14:04:59.999000+0000", tz="UTC"),
            "open": 55306.46,
            "high": 55399.68,
            "low": 55217.22,
            "close": 55388.96,
            "volume": 276.690734,
            "quote_volume": 15295597.50785806,
            "trades": 7614,
            "taker_buy_asset_volume": 145.211424,
            "taker_buy_quote_volume": 8027028.99029815,
            "returns": 0.0,
            "sma": nan,
            "upper": nan,
            "lower": nan,
        },
        {
            "open_time": Timestamp("2021-04-21 14:10:00+0000", tz="UTC"),
            "close_time": Timestamp("2021-04-21 14:09:59.999000+0000", tz="UTC"),
            "open": 55306.46,
            "high": 55399.68,
            "low": 55217.22,
            "close": 55388.96,
            "volume": 276.690734,
            "quote_volume": 15295597.50785806,
            "trades": 7614,
            "taker_buy_asset_volume": 145.211424,
            "taker_buy_quote_volume": 8027028.99029815,
            "returns": 0.0,
            "sma": 55388.96,
            "upper": 55388.96,
            "lower": 55388.96,
        },
        {
            "open_time": Timestamp("2021-04-21 14:15:00+0000", tz="UTC"),
            "close_time": Timestamp("2021-04-21 14:14:59.999000+0000", tz="UTC"),
            "open": 55306.46,
            "high": 55399.68,
            "low": 55217.22,
            "close": 55388.96,
            "volume": 276.690734,
            "quote_volume": 15295597.50785806,
            "trades": 7614,
            "taker_buy_asset_volume": 145.211424,
            "taker_buy_quote_volume": 8027028.99029815,
            "returns": 0.0,
            "sma": 55388.96,
            "upper": 55388.96,
            "lower": 55388.96,
        },
        {
            "open_time": Timestamp("2021-04-21 14:20:00+0000", tz="UTC"),
            "close_time": Timestamp("2021-04-21 14:19:59.999000+0000", tz="UTC"),
            "open": 55306.46,
            "high": 55399.68,
            "low": 55217.22,
            "close": 55388.96,
            "volume": 276.690734,
            "quote_volume": 15295597.50785806,
            "trades": 7614,
            "taker_buy_asset_volume": 145.211424,
            "taker_buy_quote_volume": 8027028.99029815,
            "returns": 0.0,
            "sma": 55388.96,
            "upper": 55388.96,
            "lower": 55388.96,
        },
        {
            "open_time": Timestamp("2021-04-21 14:25:00+0000", tz="UTC"),
            "close_time": Timestamp("2021-04-21 14:29:59.999000+0000", tz="UTC"),
            "open": 55388.95,
            "high": 55569.95,
            "low": 55388.95,
            "close": 55552.4,
            "volume": 149.363426,
            "quote_volume": 8288967.03877351,
            "trades": 5260,
            "taker_buy_asset_volume": 82.67909,
            "taker_buy_quote_volume": 4588065.23181743,
            "returns": 0.002946423556387118,
            "sma": 55443.44,
            "upper": 55726.52638398906,
            "lower": 55160.35361601094,
        },
        {
            "open_time": Timestamp("2021-04-21 14:30:00+0000", tz="UTC"),
            "close_time": Timestamp("2021-04-21 14:34:59.999000+0000", tz="UTC"),
            "open": 55550.89,
            "high": 56087.68,
            "low": 55550.89,
            "close": 55932.48,
            "volume": 692.924319,
            "quote_volume": 38726480.58078431,
            "trades": 16507,
            "taker_buy_asset_volume": 411.223017,
            "taker_buy_quote_volume": 22979821.33981915,
            "returns": 0.006818529518377586,
            "sma": 55624.613333333335,
            "upper": 56461.20232776249,
            "lower": 54788.02433890418,
        },
        {
            "open_time": Timestamp("2021-04-21 14:35:00+0000", tz="UTC"),
            "close_time": Timestamp("2021-04-21 14:39:59.999000+0000", tz="UTC"),
            "open": 55932.48,
            "high": 56333.0,
            "low": 55932.48,
            "close": 56264.93,
            "volume": 603.660118,
            "quote_volume": 33896505.6971466,
            "trades": 14656,
            "taker_buy_asset_volume": 356.915883,
            "taker_buy_quote_volume": 20037884.70780964,
            "returns": 0.005926179097336494,
            "sma": 55916.60333333333,
            "upper": 56986.1940095735,
            "lower": 54847.01265709317,
        },
        {
            "open_time": Timestamp("2021-04-21 14:40:00+0000", tz="UTC"),
            "close_time": Timestamp("2021-04-21 14:44:59.999000+0000", tz="UTC"),
            "open": 56260.11,
            "high": 56317.43,
            "low": 56118.31,
            "close": 56168.82,
            "volume": 370.500359,
            "quote_volume": 20822485.25288953,
            "trades": 8616,
            "taker_buy_asset_volume": 178.075904,
            "taker_buy_quote_volume": 10007121.78892216,
            "returns": -0.00170962942015996,
            "sma": 56122.07666666667,
            "upper": 56635.32621152748,
            "lower": 55608.82712180586,
        },
        {
            "open_time": Timestamp("2021-04-21 14:45:00+0000", tz="UTC"),
            "close_time": Timestamp("2021-04-21 14:49:59.999000+0000", tz="UTC"),
            "open": 56168.82,
            "high": 56269.99,
            "low": 56080.96,
            "close": 56191.11,
            "volume": 324.51432,
            "quote_volume": 18225087.41727558,
            "trades": 8352,
            "taker_buy_asset_volume": 145.064381,
            "taker_buy_quote_volume": 8146012.47892439,
            "returns": 0.0003967606653441501,
            "sma": 56208.28666666666,
            "upper": 56359.20072465933,
            "lower": 56057.37260867399,
        },
        {
            "open_time": Timestamp("2021-04-21 14:50:00+0000", tz="UTC"),
            "close_time": Timestamp("2021-04-21 14:54:59.999000+0000", tz="UTC"),
            "open": 56191.11,
            "high": 56200.0,
            "low": 56107.98,
            "close": 56145.0,
            "volume": 254.091606,
            "quote_volume": 14265787.89818125,
            "trades": 6455,
            "taker_buy_asset_volume": 134.1124,
            "taker_buy_quote_volume": 7529521.30623853,
            "returns": -0.0008209293091875578,
            "sma": 56168.31,
            "upper": 56237.48769076675,
            "lower": 56099.132309233246,
        },
        {
            "open_time": Timestamp("2021-04-21 14:55:00+0000", tz="UTC"),
            "close_time": Timestamp("2021-04-21 14:59:59.999000+0000", tz="UTC"),
            "open": 56145.0,
            "high": 56211.7,
            "low": 56106.97,
            "close": 56182.11,
            "volume": 270.145731,
            "quote_volume": 15171017.18758856,
            "trades": 7707,
            "taker_buy_asset_volume": 168.231118,
            "taker_buy_quote_volume": 9447425.0774598,
            "returns": 0.0006607487960858385,
            "sma": 56172.74,
            "upper": 56246.06411813189,
            "lower": 56099.4158818681,
        },
        {
            "open_time": Timestamp("2021-04-21 15:00:00+0000", tz="UTC"),
            "close_time": Timestamp("2021-04-21 15:04:59.999000+0000", tz="UTC"),
            "open": 56182.12,
            "high": 56299.78,
            "low": 56172.09,
            "close": 56289.89,
            "volume": 298.797415,
            "quote_volume": 16804824.55255641,
            "trades": 9000,
            "taker_buy_asset_volume": 139.83665,
            "taker_buy_quote_volume": 7864202.02549528,
            "returns": 0.0019165664875115606,
            "sma": 56205.666666666664,
            "upper": 56431.45459570587,
            "lower": 55979.878737627456,
        },
    ]
)

expected_signal = None