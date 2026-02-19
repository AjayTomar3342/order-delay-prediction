""" Using classes:
a.) Use when something has state which changes
b.) Use when something is reused
c.) Use when something has a lifecycle

In ML, make a class for data Ingestion, feature engineering, model wrapper class, training pipeline, fastapi class.
"""

""" Log when: 
a.) You need to show pipeline steps execution: data Ingestion, Feature engineering, Model training, Prediction results.
b.) You need to define errors & exceptions.
c.) Each of the step in a.) should include multiple important logging details. 
d.) Model details such as feature count, hyperparameters, training metrics, model files should be logged using mlflow. 
"""

""" Unit Tests: 
a.) Need to use assert. (x = 5, assert x = 5, "x soll 5") works. (x = 5, assert x = 10, x soll 10) does not work.
b.) Example:
Original Function:
            def add(a, b):
                return a + b
Test:
            assert add(2, 3) == 5
-- If someone change add function's working, this test fails

c.) Test if a column exists, values are correct, no unexpected NaN's, feature count, o/p shape, o/p type is stable
"""

""" Exception Handling: 
a.) Error types possible are ValueError (Invalid Input), TypeError (Wrong Type), FileNotFoundError, RuntimeError
b.) Example:
            def calculate_delay_days(order_date, delivery_date):
                if delivery_date < order_date:
                    raise ValueError("delivery_date cannot be before order_date")
            
                return (delivery_date - order_date).days
"""

""" Unit Test Vs. Exception Handling (E.g., Functions I/P, O/P Test): 
a.) Use Exception Handling when you want to stop execution and continuing would produce wrong results.
b.) With unit test, you verify defined behavior.

Exception Handling:
            def calculate_delay_days(order_date, delivery_date):
                if delivery_date < order_date:
                    raise ValueError("delivery_date cannot be before order_date")
            
                return (delivery_date - order_date).days

Unit Test:
            import pytest
            from datetime import datetime, timedelta
            
            def test_calculate_delay_days_valid():
                d1 = datetime(2024, 1, 1)
                d2 = datetime(2024, 1, 4)
            
                result = calculate_delay_days(d1, d2)
            
                assert result == 3
"""

