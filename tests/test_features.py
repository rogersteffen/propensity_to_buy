import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pipeline')))

# Import the Features class
from features import Features

def test_time_slice_feature_sql():
    # Example inputs
    offset_length = 7
    offset_name = "week"
    end_interval = 2
    feature_end = "2025-01-01"
    start_interval = 1

    # Expected output
    expected_output = """
                ,SUM(CASE WHEN t.t_dat > DATE '2025-01-01' - INTERVAL (7*1) DAY AND t.t_dat <= DATE '2025-01-01' - INTERVAL (7*(1 - 1)) DAY THEN 1 ELSE 0 END)
                    as t_count_week_1
                ,COUNT(DISTINCT CASE WHEN t.t_dat > DATE '2025-01-01' - INTERVAL (7*1) DAY AND t.t_dat <= DATE '2025-01-01' - INTERVAL (7*(1 - 1)) DAY THEN t.t_dat ELSE NULL END)
                    as ti_count_week_1
                ,SUM(CASE WHEN t.t_dat > DATE '2025-01-01' - INTERVAL (7*1) DAY AND t.t_dat <= DATE '2025-01-01' - INTERVAL (7*(1 - 1)) DAY THEN 590*price ELSE 0 END)
                    as revenue_week_1
        

                ,SUM(CASE WHEN t.t_dat > DATE '2025-01-01' - INTERVAL (7*2) DAY AND t.t_dat <= DATE '2025-01-01' - INTERVAL (7*(2 - 1)) DAY THEN 1 ELSE 0 END)
                    as t_count_week_2
                ,COUNT(DISTINCT CASE WHEN t.t_dat > DATE '2025-01-01' - INTERVAL (7*2) DAY AND t.t_dat <= DATE '2025-01-01' - INTERVAL (7*(2 - 1)) DAY THEN t.t_dat ELSE NULL END)
                    as ti_count_week_2
                ,SUM(CASE WHEN t.t_dat > DATE '2025-01-01' - INTERVAL (7*2) DAY AND t.t_dat <= DATE '2025-01-01' - INTERVAL (7*(2 - 1)) DAY THEN 590*price ELSE 0 END)
                    as revenue_week_2
        """

    # Actual output
    actual_output = Features.time_slice_feature_sql(offset_length, offset_name, end_interval, feature_end, start_interval)

    # Assert
    assert actual_output.strip() == expected_output.strip(), "Test failed: Outputs do not match."
