# tests/unit/test_features.py
import itertools
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from data_loading.features.time_features import CyclicalTimeFeature, HourOfWeekFeature, OneHotTimeFeature, WorkdayFeature


@pytest.fixture
def sample_datetime():
    return pd.Timestamp('2024-01-01 12:00:00')  # Monday at noon

def test_cyclical_time_feature():
    feature = CyclicalTimeFeature(24)  # 24-hour cycle
    result = feature.generate(pd.Series([12]))  # noon
    assert len(result) == 2  # sin and cos components
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)
    assert -1 <= result[0] <= 1  # sin bounds
    assert -1 <= result[1] <= 1  # cos bounds
    assert feature.get_feature_dim() == 2

def test_one_hot_time_feature():
    feature = OneHotTimeFeature(24)  # 24-hour encoding
    result = feature.generate(12)  # noon
    assert len(result) == 24
    assert sum(result) == 1.0  # only one hot position
    assert result[12] == 1.0  # correct position is hot
    assert feature.get_feature_dim() == 24
    assert all(isinstance(x, float) for x in result)

def test_hour_of_week_feature(sample_datetime):
    feature = HourOfWeekFeature()
    result = feature.generate(sample_datetime)
    assert len(result) == 1
    assert isinstance(result[0], float)

    # Calculate expected hour of week (Monday at noon)
    # Monday is 0, so 12 hours into Monday = 12
    expected_hour = float(12)  # Must be float
    assert result[0] == expected_hour

    # Test a different day
    wednesday_noon = pd.Timestamp('2024-01-03 12:00:00')  # Wednesday at noon
    result_wed = feature.generate(wednesday_noon)
    # Wednesday is 2, so 12 hours into Wednesday = 2 * 24 + 12 = 60
    expected_hour_wed = float(2 * 24 + 12)
    assert result_wed[0] == expected_hour_wed

    assert feature.get_feature_dim() == 1

def test_hour_of_week_feature_bounds(sample_datetime):
    feature = HourOfWeekFeature()
    # Test all days of the week
    for day, hour in itertools.product(range(7), range(24)):
        test_date = sample_datetime + pd.Timedelta(days=day, hours=hour-12)
        result = feature.generate(test_date)
        assert 0 <= result[0] <= 167  # 7 days * 24 hours - 1
        assert isinstance(result[0], float)
        expected = float(day * 24 + hour)
        assert result[0] == expected

def test_workday_feature(sample_datetime):
    # sourcery skip: extract-duplicate-method
    feature = WorkdayFeature()
    result = feature.generate(sample_datetime)
    assert len(result) == 5  # workday, holiday, prev_workday, next_workday, christmas
    assert all(isinstance(x, float) for x in result)
    assert all(x in [0.0, 1.0] for x in result)  # binary features
    assert feature.get_feature_dim() == 5

    # Test specific workday logic
    monday = pd.Timestamp('2024-01-01 12:00:00')  # New Year's Day (holiday)
    result_monday = feature.generate(monday)
    assert result_monday[0] == 0.0  # not a workday (holiday)
    assert result_monday[1] == 1.0  # is a holiday

    regular_tuesday = pd.Timestamp('2024-01-02 12:00:00')  # Regular Tuesday
    result_tuesday = feature.generate(regular_tuesday)
    assert result_tuesday[0] == 1.0  # is a workday
    assert result_tuesday[1] == 0.0  # not a holiday

def test_cyclical_time_feature_error_handling():
    feature = CyclicalTimeFeature(24)
    with pytest.raises(ValueError):
        feature.generate(pd.Series([12, 13]))  # Multiple values should raise error

def test_workday_feature_numpy_datetime(sample_datetime):
    feature = WorkdayFeature()
    np_datetime = np.datetime64(sample_datetime)
    result = feature.generate(np_datetime)
    assert len(result) == 5
    assert all(isinstance(x, float) for x in result)

    # Test conversion handling
    pd_result = feature.generate(sample_datetime)
    np_result = feature.generate(np_datetime)
    assert np.array_equal(pd_result, np_result)  # Results should be identical regardless of input type

def test_cyclical_time_feature_full_cycle():
    feature = CyclicalTimeFeature(24)  # 24-hour cycle
    # Test full cycle of hours
    for hour in range(24):
        result = feature.generate(pd.Series([hour]))
        assert len(result) == 2
        assert -1 <= result[0] <= 1
        assert -1 <= result[1] <= 1
        # Test that 0 and 24 give the same result
        if hour == 0:
            result_24 = feature.generate(pd.Series([24]))
            assert np.allclose(result, result_24)