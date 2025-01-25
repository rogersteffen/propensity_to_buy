from datetime import date, timedelta

import polars as pl
import duckdb
from functools import reduce


class Features:

    BASE_FEATURE_QUERY = '''
        SELECT
            t.customer_id
            {feature_sql}
        FROM transactions t
        INNER JOIN customers c ON t.customer_id = c.customer_id
        WHERE t.t_dat > DATE '{feature_start}' and t.t_dat <= DATE '{feature_end}' 
        GROUP BY t.customer_id
    '''

    def __init__(self, end_date, feature_duration=365, response_duration=0, additional_offset=0):
        self.end_date = end_date
        self.feature_duration = feature_duration
        self.response_duration = response_duration
        self.additional_offset = additional_offset # used if backtesting after training
        self.response_end = self.end_date - timedelta(days=self.additional_offset)
        self.feature_end = self.response_end - timedelta(days=self.response_duration)
        self.response_start = self.feature_end # yes both the same ... must be careful!
        self.feature_start = self.feature_end - timedelta(days=self.feature_duration)

    @staticmethod
    def run_query(duckdb_conn, query: str) -> pl.DataFrame:
        arrow_table = duckdb_conn.execute(query).fetch_arrow_table()
        # Convert the Arrow Table to a Polars DataFrame
        return pl.from_arrow(arrow_table)

    @staticmethod
    def time_slice_feature_sql(offset_length: int, offset_name: str, end_interval, feature_end='{feature_end}',
                               start_interval=1):
        '''
        This is a helper utility to general bocks of SQL.   This can be run manually or incorporated into the full query creation.
        :param offset_length: Number of days
        :param offset_name: week, month, quarter, half ... and if it will be use with other filters, then online_week, instore_week
        :param end_interval: How many time slices needed + start_interval - 1
        :param feature_end: Likely keep the default value as this is parameterized for use for time shifting
        :param start_interval: Default is 1 ... can change if you want, for example, only the second half of a year
        :return:
        '''
        block = """
                ,SUM(CASE WHEN t.t_dat > DATE '{feature_end}' - INTERVAL ({offset_length}*{i}) DAY AND t.t_dat <= DATE '{feature_end}' - INTERVAL ({offset_length}*({i} - 1)) DAY THEN 1 ELSE 0 END)
                    as t_count_{offset_name}_{i}
                ,COUNT(DISTINCT CASE WHEN t.t_dat > DATE '{feature_end}' - INTERVAL ({offset_length}*{i}) DAY AND t.t_dat <= DATE '{feature_end}' - INTERVAL ({offset_length}*({i} - 1)) DAY THEN t.t_dat ELSE NULL END)
                    as ti_count_{offset_name}_{i}
                ,SUM(CASE WHEN t.t_dat > DATE '{feature_end}' - INTERVAL ({offset_length}*{i}) DAY AND t.t_dat <= DATE '{feature_end}' - INTERVAL ({offset_length}*({i} - 1)) DAY THEN 590*price ELSE 0 END)
                    as revenue_{offset_name}_{i}
        """
        result = []
        for i in range(start_interval, end_interval + 1):
            result.append(block.format(offset_length=offset_length, offset_name=offset_name, feature_end=feature_end, i=i))

        return "\n".join(result)

    @staticmethod
    def get_base_features() -> pl.DataFrame:
        pass

    def get_response_label(self, duckdb_session) -> pl.DataFrame:
        response_query = """
                SELECT
                    t.customer_id,
                    MAX(
                        CASE
                            WHEN t.t_dat > DATE '{response_start}'  AND t.t_dat <= DATE '{response_end}'  THEN 1
                            ELSE 0
                        END
                    ) AS label
                FROM transactions t
                INNER JOIN customers c ON c.customer_id = t.customer_id
                GROUP BY t.customer_id
        """

        response_query = response_query.format(response_start=self.response_start, response_end=self.response_end)
        return Features.run_query(duckdb_session, response_query)


    def get_time_sliced_no_overlap(self, duckdb_session) -> pl.DataFrame:
        feature_sql = Features.time_slice_feature_sql(offset_length=28*3, offset_name="quarter", end_interval=4,
                feature_end=self.feature_end, start_interval=1)
        earliest_month = Features.time_slice_feature_sql(offset_length=28, offset_name="month", end_interval=13,
                feature_end=self.feature_end, start_interval=13)

        inner_sql = "\n".join([feature_sql, earliest_month])

        complete_sql = Features.BASE_FEATURE_QUERY.format(feature_sql=inner_sql, feature_start=self.feature_start, feature_end=self.feature_end)

        return Features.run_query(duckdb_session, complete_sql)

    @staticmethod
    def get_time_sliced_overlap() -> pl.DataFrame:
        pass


    def get_customer_features(self, duckdb_session) -> pl.DataFrame:

        customer_sql = '''
            ,ROUND(ROUND(590*SUM(price))/COUNT(DISTINCT t.t_dat)) as aov
            ,MAX(CASE WHEN COALESCE(c.active,0) = 1 THEN 1 ELSE 0 END) AS customer_active
            ,MAX(CASE WHEN COALESCE(c.fashion_news_frequency, 'Empty') in ('Monthly','Reqularly') THEN 1 ELSE 0 END) AS customer_fashion_news_frequency
            ,MAX(CASE WHEN COALESCE(c.FN,0) = 1 THEN 1 ELSE 0 END) AS customer_fn
            ,ROUND(SUM(CASE WHEN sales_channel_id = 1 THEN 1 ELSE 0 END)/COUNT(1),0) AS primary_sales_channel_01
            ,ROUND(SUM(CASE WHEN sales_channel_id = 2 THEN 1 ELSE 0  END)/COUNT(1),0) AS primary_sales_channel_02
            ,MAX(COALESCE(c.age,-1)) as age
        '''

        complete_sql = Features.BASE_FEATURE_QUERY.format(feature_sql=customer_sql, feature_start=self.feature_start,
                                                          feature_end=self.feature_end)

        return Features.run_query(duckdb_session, complete_sql)

    @staticmethod
    def get_all_features_and_response() -> pl.DataFrame:

        # Sample DataFrames
        df1 = pl.DataFrame({"id": [1, 2, 3], "value1": [10, 20, 30]})
        df2 = pl.DataFrame({"id": [1, 2, 3], "value2": [100, 200, 300]})
        df3 = pl.DataFrame({"id": [1, 2, 3], "value3": [1000, 2000, 3000]})

        # List of DataFrames to join
        dfs = [df1, df2, df3]

        # Joining multiple DataFrames on the same column
        result = reduce(lambda left, right: left.join(right, on="id", how="inner"), dfs)

        print(result)
        pass

    @staticmethod
    def get_season_features() -> pl.DataFrame:
        pass

    @staticmethod
    def get_product_features() -> pl.DataFrame:
        pass





#
# class QueryConstants:
#     end_date=date(2020,9,22)
#     feature_duration=365
#     label_duration=30
#     backtest_duration=30
#     label_end =  end_date - timedelta(days=backtest_duration)
#     feature_end = label_end - timedelta(days=label_duration)
#     feature_start = feature_end - timedelta(days=feature_duration)
#
# response_query = """
#         SELECT
#             t.customer_id,
#             MAX(
#                 CASE
#                     WHEN t.t_dat > DATE '{response_start}'  AND t.t_dat <= DATE '{response_end}'  THEN 1
#                     ELSE 0
#                 END
#             ) AS label
#         FROM transactions t
#         INNER JOIN customers c ON c.customer_id = t.customer_id
#         GROUP BY t.customer_id
# """
#
#
# feature_query = '''
# SELECT *
#     ,SQRT(t_count_quarter_1*t_count_quarter_2) as q1_x_q2
#     ,SQRT(t_count_quarter_1*t_count_quarter_3) as q1_x_q3
#     ,SQRT(t_count_quarter_1*t_count_quarter_4) as q1_x_q4
#     ,SQRT(t_count_quarter_2*t_count_quarter_3) as q2_x_q3
#     ,SQRT(t_count_quarter_2*t_count_quarter_4) as q2_x_q4
#     ,SQRT(t_count_quarter_3*t_count_quarter_4) as q3_x_q4
#     ,POWER(t_count_quarter_1*t_count_quarter_2*t_count_quarter_3*t_count_quarter_4,0.25) as q1_x_q2_x_q3_x_q4
# FROM (
#     SELECT
#         t.customer_id
#
#         ,SUM(CASE WHEN t.t_dat > DATE '{feature_end}' - INTERVAL (84*1) DAY AND t.t_dat <= DATE '{feature_end}' - INTERVAL (84*(1 - 1)) DAY THEN 1 ELSE 0 END)
#             as t_count_quarter_1
#         ,COUNT(DISTINCT CASE WHEN t.t_dat > DATE '{feature_end}' - INTERVAL (84*1) DAY AND t.t_dat <= DATE '{feature_end}' - INTERVAL (84*(1 - 1)) DAY THEN t.t_dat ELSE NULL END)
#             as ti_count_quarter_1
#         ,SUM(CASE WHEN t.t_dat > DATE '{feature_end}' - INTERVAL (84*1) DAY AND t.t_dat <= DATE '{feature_end}' - INTERVAL (84*(1 - 1)) DAY THEN 590*price ELSE 0 END)
#             as revenue_quarter_1
#
#
#         ,SUM(CASE WHEN t.t_dat > DATE '{feature_end}' - INTERVAL (84*2) DAY AND t.t_dat <= DATE '{feature_end}' - INTERVAL (84*(2 - 1)) DAY THEN 1 ELSE 0 END)
#             as t_count_quarter_2
#         ,COUNT(DISTINCT CASE WHEN t.t_dat > DATE '{feature_end}' - INTERVAL (84*2) DAY AND t.t_dat <= DATE '{feature_end}' - INTERVAL (84*(2 - 1)) DAY THEN t.t_dat ELSE NULL END)
#             as ti_count_quarter_2
#         ,SUM(CASE WHEN t.t_dat > DATE '{feature_end}' - INTERVAL (84*2) DAY AND t.t_dat <= DATE '{feature_end}' - INTERVAL (84*(2 - 1)) DAY THEN 590*price ELSE 0 END)
#             as revenue_quarter_2
#
#
#         ,SUM(CASE WHEN t.t_dat > DATE '{feature_end}' - INTERVAL (84*3) DAY AND t.t_dat <= DATE '{feature_end}' - INTERVAL (84*(3 - 1)) DAY THEN 1 ELSE 0 END)
#             as t_count_quarter_3
#         ,COUNT(DISTINCT CASE WHEN t.t_dat > DATE '{feature_end}' - INTERVAL (84*3) DAY AND t.t_dat <= DATE '{feature_end}' - INTERVAL (84*(3 - 1)) DAY THEN t.t_dat ELSE NULL END)
#             as ti_count_quarter_3
#         ,SUM(CASE WHEN t.t_dat > DATE '{feature_end}' - INTERVAL (84*3) DAY AND t.t_dat <= DATE '{feature_end}' - INTERVAL (84*(3 - 1)) DAY THEN 590*price ELSE 0 END)
#             as revenue_quarter_3
#
#
#         ,SUM(CASE WHEN t.t_dat > DATE '{feature_end}' - INTERVAL (84*4) DAY AND t.t_dat <= DATE '{feature_end}' - INTERVAL (84*(4 - 1)) DAY THEN 1 ELSE 0 END)
#             as t_count_quarter_4
#         ,COUNT(DISTINCT CASE WHEN t.t_dat > DATE '{feature_end}' - INTERVAL (84*4) DAY AND t.t_dat <= DATE '{feature_end}' - INTERVAL (84*(4 - 1)) DAY THEN t.t_dat ELSE NULL END)
#             as ti_count_quarter_4
#         ,SUM(CASE WHEN t.t_dat > DATE '{feature_end}' - INTERVAL (84*4) DAY AND t.t_dat <= DATE '{feature_end}' - INTERVAL (84*(4 - 1)) DAY THEN 590*price ELSE 0 END)
#             as revenue_quarter_4
#
#
#         ,SUM(CASE WHEN t.t_dat > DATE '{feature_end}' - INTERVAL (28*13) DAY AND t.t_dat <= DATE '{feature_end}' - INTERVAL (28*(13 - 1)) DAY THEN 1 ELSE 0 END)
#             as t_count_month_13
#         ,COUNT(DISTINCT CASE WHEN t.t_dat > DATE '{feature_end}' - INTERVAL (28*13) DAY AND t.t_dat <= DATE '{feature_end}' - INTERVAL (28*(13 - 1)) DAY THEN t.t_dat ELSE NULL END)
#             as ti_count_month_13
#         ,SUM(CASE WHEN t.t_dat > DATE '{feature_end}' - INTERVAL (28*13) DAY AND t.t_dat <= DATE '{feature_end}' - INTERVAL (28*(13 - 1)) DAY THEN 590*price ELSE 0 END)
#             as revenue_month_13
#
#
#         ,COUNT(1) total_transaction_items
#         ,MAX(t.t_dat) - DATE '{feature_end}' as days_since_last
#         --TODO: Perhaps make the channel data driven
#         ,MAX(CASE WHEN sales_channel_id = 1 THEN t.t_dat ELSE DATE '{feature_start}' END) - DATE '{feature_end}' as days_since_last_channel_1
#         ,MAX(CASE WHEN sales_channel_id = 2 THEN t.t_dat ELSE DATE '{feature_start}' END) - DATE '{feature_end}' as days_since_last_channel_2
#         ,ROUND(590*SUM(price)) as total_revenue
#         ,COUNT(DISTINCT t.t_dat) as total_transactions
#         ,ROUND(ROUND(590*SUM(price))/COUNT(DISTINCT t.t_dat)) as aov
#         ,SUM(CASE WHEN sales_channel_id = 1 THEN 1 ELSE 0 END) AS sales_channel_01
#         ,SUM(CASE WHEN sales_channel_id = 2 THEN 1 ELSE 0  END) AS sales_channel_02
#         ,MAX(CASE WHEN COALESCE(c.FN,0) = 1 THEN 1 ELSE 0 END) AS customer_fn
#         ,MAX(CASE WHEN COALESCE(c.active,0) = 1 THEN 1 ELSE 0 END) AS customer_active
#         ,MAX(CASE WHEN COALESCE(c.fashion_news_frequency, 'Empty') in ('Monthly','Reqularly') THEN 1 ELSE 0 END) AS customer_fashion_news_frequency
#
#     FROM transactions t
#     INNER JOIN customers c ON t.customer_id = c.customer_id
#     WHERE t.t_dat <= DATE '{feature_end}' and t.t_dat > DATE '{feature_start}'
#     --and customer_id = '00d40c65c316c02eac7fb0c628afbf57d616eed4b08f69ecfc115ca643a308af'
#     GROUP BY t.customer_id
# ) x
# '''
#
# label_query = response_query.format(response_start=QueryConstants.feature_end, response_end=QueryConstants.label_end)
# backtest_response_query = response_query.format(response_start=QueryConstants.label_end, response_end=QueryConstants.end_date)
#
# arrow_table = dataset.duckdb_conn.execute(label_query).fetch_arrow_table()
# label_df = pl.from_arrow(arrow_table)
#
# arrow_table = dataset.duckdb_conn.execute(backtest_response_query).fetch_arrow_table()
# backtest_response_df = pl.from_arrow(arrow_table)
#
# query = feature_query.format(**vars(QueryConstants))
# arrow_table = dataset.duckdb_conn.execute(query).fetch_arrow_table()
# feature_df = pl.from_arrow(arrow_table)
#
#
# display(feature_df)