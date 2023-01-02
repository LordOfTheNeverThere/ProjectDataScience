#### aggregation

def aggregate_by(data: Series, index_var: str, period: str):
    index = data.index.to_period(period)
    agg_df = data.copy().groupby(index).mean()
    agg_df[index_var] = index.drop_duplicates().to_timestamp()
    agg_df.set_index(index_var, drop=True, inplace=True)
    return agg_df

# figure(figsize=(3*HEIGHT, HEIGHT))
# agg_df = aggregate_by(data, 'timestamp', 'D')
# plot_series(agg_df, title='Daily values', x_label='timestamp', y_label='value')
# xticks(rotation = 45)
# show()

# figure(figsize=(3*HEIGHT, HEIGHT))
# agg_df = aggregate_by(data, 'timestamp', 'W')
# plot_series(agg_df, title='Weekly values', x_label='timestamp', y_label='value')
# xticks(rotation = 45)
# show()

# figure(figsize=(3*HEIGHT, HEIGHT))
# agg_df = aggregate_by(data, 'timestamp', 'M')
# plot_series(agg_df, title='Monthly values', x_label='timestamp', y_label='value')
# xticks(rotation = 45)
# show()

## multivaried series

figure(figsize=(3*HEIGHT, HEIGHT))
agg_multi_df = aggregate_by(data_multi, index_multi, 'D')
plot_series(agg_multi_df[target_multi], title='Appliances - Daily values', x_label='timestamp', y_label='value')
#plot_series(agg_multi_df['lights'])
xticks(rotation = 45)
savefig('images/transformation/set2_data_aggregation_day.png')
show()

figure(figsize=(3*HEIGHT, HEIGHT))
agg_multi_df = aggregate_by(data_multi, index_multi, 'W')
plot_series(agg_multi_df[target_multi], title='Appliances - Weekly values', x_label='timestamp', y_label='value')
#plot_series(agg_multi_df['lights'])
xticks(rotation = 45)
savefig('images/transformation/set2_data_aggregation_week.png')
show()

figure(figsize=(3*HEIGHT, HEIGHT))
agg_multi_df = aggregate_by(data_multi, index_multi, 'M')
plot_series(agg_multi_df[target_multi], title='Appliances - Monthly values', x_label='timestamp', y_label='value')
#plot_series(agg_multi_df['lights'], x_label='timestamp', y_label='value')
xticks(rotation = 45)
savefig('images/transformation/set2_data_aggregation_month.png')
show()

