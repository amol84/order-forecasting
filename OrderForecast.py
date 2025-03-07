
import pandas as pd

from prophet.plot import plot_plotly, plot_components_plotly
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

WEEKLY_END_DATES_CSV = "/Users/amaggarw/Downloads/weekly_end_dates.csv"


def forecast_orders():
    orders_df = pd.read_csv(WEEKLY_END_DATES_CSV, parse_dates=['WeekEndDate'])
    orders_df.rename(columns={'WeekEndDate': 'ds', 'OrderCount': 'y'}, inplace=True)

    max_orders_limit = 1000000
    orders_df['cap'] = max_orders_limit
    print(orders_df)

    holidays = pd.DataFrame({
        'holiday': ['thanksgiving', 'cyber_monday', 'memorial_1', 'memorial_2'] *2,
        'ds': pd.to_datetime([ '2023-11-24', '2023-12-01','2023-05-26','2023-06-02',
                               '2024-11-29', '2024-12-06','2024-05-24', '2024-05-31']),
        'lower_window': 0,
        'upper_window': 0,
    })

    # multiplicative seasonality is effect of season becomes more prominent with increase in orders
    model = Prophet(yearly_seasonality= True, seasonality_mode = 'multiplicative', holidays = holidays, growth = 'logistic')

    # run test data on the model
    model.fit(orders_df)

    # predict aggregate orders for each week for next 52 weeks with week ending on friday
    future_df = model.make_future_dataframe(periods=52, freq='W-FRI')
    #add max orders to future dataframe
    future_df['cap'] = max_orders_limit
    #predict the future
    predicted_data = model.predict(future_df)

    #plot data
    #model.plot(predicted_data)
    # model.plot_components(predicted_data)
    fig = plot_plotly(model, predicted_data)
    fig.show()

    # For components plot
    components_fig = plot_components_plotly(model, predicted_data)
    components_fig.show()

    cross_validation_ds = cross_validation(model, initial='60 W', period='26 W', horizon='42 W')

    metrics = performance_metrics(cross_validation_ds, metrics = ['mape', 'rmse', 'mae'])

    #mape - mean absolute percent error  -> measure percentage error in order count for forecasted data
    #rmse - root mean square error -> measure overall deviation - > penalize large errors and work well for detecting outliers
    #mae - mean absolute error - 1/n sum(y-yhat) ->predict sale volume keeping actual sale nos.
    final_ds = cross_validation_ds[['ds', 'y', 'yhat']].merge(metrics, left_index= True, right_index = True)
    print(final_ds[['ds', 'y', 'yhat', 'mape', 'rmse', 'mae']])


if __name__ == "__main__":
    forecast_orders()