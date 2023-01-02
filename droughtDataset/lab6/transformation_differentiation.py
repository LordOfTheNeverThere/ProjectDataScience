#### differentiation

# diff_df = data.diff()
# figure(figsize=(3*HEIGHT, HEIGHT))
# plot_series(diff_df, title='Differentiation', x_label='timestamp', y_label='value')
# xticks(rotation = 45)
# show()

## multivaried series

diff_df_multi = data_multi.diff()
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(diff_df_multi[target_multi], title='Appliances - Differentiation', x_label=index_multi, y_label='value')
#plot_series(diff_df_multi['lights'])
xticks(rotation = 45)
savefig('images/transformation/set2_data_differentiation.png')
show()