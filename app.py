import streamlit as st
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.set_page_config(page_title="Inventory AutoML + Viz", layout="wide")

# Optuna
def optimize_model(X, y, model_type, depth):
    n_trials = {"Быстрое": 5, "Обычное": 20, "Долгое": 50}[depth]
    tscv = TimeSeriesSplit(n_splits=3)
    def objective(trial):
        if model_type == "GB":
            model = GradientBoostingRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1),
                max_depth=trial.suggest_int("max_depth", 3, 8),
                random_state=42
            )
        else:
            model = Ridge(alpha=trial.suggest_float("alpha", 0.1, 10.0),
                        random_state=42)
        
        scores = []
        for train_idx, val_idx in tscv.split(X):
            model.fit(X[train_idx], y[train_idx])
            scores.append(np.sqrt(mean_squared_error(y[val_idx], model.predict(X[val_idx]))))
        return np.mean(scores)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

@st.cache_data
# Main function
def run_full_analysis_with_plots(input_df, prices_df, test_size, depth):
    summary_results = []
    plot_data = {}

    df_clean = input_df.copy()
    df_clean['Product'] = df_clean['Product'].astype(str).str.strip()
    
    # Dates aggregation
    df_clean = df_clean.groupby(['Product', 'Date']).agg({
        'Demand': 'sum',
        'Stock': 'last' 
    }).reset_index()

    for prod in df_clean['Product'].unique():
        prod_df = df_clean[df_clean['Product'] == prod].sort_values('Date').copy()
        lag_cols = []
        for lag in range(1, 7):
            col_name = f'Demand_Lag{lag}'
            prod_df[col_name] = prod_df['Demand'].shift(lag)
            lag_cols.append(col_name)
        
        prod_df = prod_df.dropna(subset=lag_cols).reset_index(drop=True)

        price_row = prices_df[prices_df['Product'] == prod]
        if price_row.empty or len(prod_df) < 10: continue
        p = price_row.iloc[0]
        
        # Preservation
        A = p.get('Preservation_A', 1.0)
        
        X = prod_df[lag_cols].values
        y = prod_df['Demand'].values
        dates = prod_df['Date']
        
        split_idx = int(len(prod_df) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        test_dates = dates[split_idx:]
        
        # Training
        best_params_gb = optimize_model(X_train, y_train, "GB", depth)
        best_params_ridge = optimize_model(X_train, y_train, "Ridge", depth)

        

        model_gb = GradientBoostingRegressor(**best_params_gb, random_state=42).fit(X_train, y_train)
        model_ridge = Ridge(**best_params_ridge, random_state=42).fit(X_train, y_train)

        # Interval windows
        best_w = 12
        min_err = float('inf')
        for w in [3, 6, 12, 18]:
            val = y_train[-w:].max()
            err = np.sqrt(mean_squared_error(y_train[-3:], [val]*3))
            if err < min_err: min_err = err; best_w = w

        preds_gb = model_gb.predict(X_test)
        preds_ridge = model_ridge.predict(X_test)
        preds_ch = np.full(len(y_test), y_train[-best_w:].max())

        # Model configs
        model_configs = {
            'GB': f"Глубина: {best_params_gb['max_depth']}, Деревьев: {best_params_gb['n_estimators']}",
            'Ridge': f"Alpha: {round(best_params_ridge['alpha'], 4)}",
            f'Interval_{best_w}m': f"Окно: {best_w} мес."
        }
        # Profit simulation
        def get_metrics(strategy_name, model_obj=None, X_train=None, y_train=None, X_test=None, y_test=None):
            # RMSE on train
            if strategy_name == 'Interval':
                train_preds = pd.Series(y_train).rolling(window=best_w, min_periods=1).max().shift(1).fillna(y_train[0]).values
            else:
                train_preds = model_obj.predict(X_train)
            
            rmse_train = np.sqrt(mean_squared_error(y_train, train_preds))

            delay = int(p.get('Delivery_Delay', 0))

            current_stock = prod_df['Stock'].iloc[split_idx]
            prof, short = 0, 0
            
            # Window initialization
            history = list(y_train[-best_w:])

            in_transit_queue = [] 
            
            # Dynamic window predictions
            dynamic_preds = []
            
            fact_stocks = []
            fictional_stocks = []
            orders_volume =[]
            for i in range(len(y_test)):
                actual_d = y_test[i]
                
                arrived_goods = 0

                still_in_transit = []
                for order_qty, arrival_month in in_transit_queue:
                    if arrival_month <= i:
                        arrived_goods += order_qty
                    else:
                        still_in_transit.append((order_qty, arrival_month))
                in_transit_queue = still_in_transit

                current_stock += arrived_goods

                # Target level
                if strategy_name.startswith('Interval'):
                    # Adaptive d_max based on window
                    current_d_max = max(history)
                    target_level = current_d_max
                elif strategy_name == 'GB':
                    target_level = preds_gb[i]
                elif strategy_name == 'Ridge':
                    target_level = preds_ridge[i]
                
                dynamic_preds.append(target_level)

                # Orders and sells
                goods_in_transit = sum(order_qty for order_qty, _ in in_transit_queue)
                fictional_stock = current_stock + goods_in_transit
                
                fact_stocks.append(current_stock)
                fictional_stocks.append(fictional_stock)

                # Order to target level
                u = max(0, target_level - fictional_stock)
                orders_volume.append(u)

                if u > 0:
                    in_transit_queue.append((u, i + delay))
                    buy_cost = u * p['Buy_Price']
                else:
                    buy_cost = 0

                available_for_sale = current_stock 
                sold = min(actual_d, available_for_sale)
                
                # Preservation
                # Remains after sales
                remains_after_sales = max(0, available_for_sale - actual_d)
                
                # Spoiled units
                spoiled_units = remains_after_sales * (1 - A)
                
                # Profit and shortage
                revenue = sold * p['Sell_Price']
                buy_cost = u * p['Buy_Price']
                storage_cost = remains_after_sales * p['Storage_Price']
                spoiled_cost = spoiled_units * p['Utilization_Cost']
                
                prof += (revenue - buy_cost - storage_cost - spoiled_cost)
                short += max(0, actual_d - available_for_sale)

                # Stock update
                current_stock = remains_after_sales * A
                
                # Window update
                history.append(actual_d)
                history.pop(0)

            # RMSE on test
            rmse_test = np.sqrt(mean_squared_error(y_test, dynamic_preds))
            
            return prof, short, dynamic_preds, rmse_test, rmse_train, fact_stocks, fictional_stocks, orders_volume

        # Results visualization
        plot_data[prod] = pd.DataFrame({
            'Date': dates, 
            'Actual': y,
            'Is_Test': [False] * len(y_train) + [True] * len(y_test)
        })

        for m_name in ['GB', 'Ridge', f'Interval_{best_w}m']:
            if m_name == 'GB':
                curr_model = model_gb
                internal_name = 'GB'
            elif m_name == 'Ridge':
                curr_model = model_ridge
                internal_name = 'Ridge'
            else:
                curr_model = None
                internal_name = 'Interval'
                
            pr, sh, d_preds, rmse_t, rmse_tr, f_st, fic_st, ord_v = get_metrics(
                internal_name, 
                model_obj=curr_model, 
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test
            )
            summary_results.append({
                'Продукт': prod, 
                'Стратегия': m_name, 
                'Прибыль (руб.)': round(pr, 2), 
                'Дефицит (ед.)': round (sh, 2),
                'RMSE на обучении (ед.)': round(rmse_tr, 2),
                'RMSE на тесте (ед.)': round(rmse_t, 2),
                'Параметры': model_configs.get(m_name, "N/A")
            })
            
            full_preds = [np.nan] * len(y_train) + list(d_preds)
            full_f_st = [np.nan] * len(y_train) + list(f_st)
            full_fic_st = [np.nan] * len(y_train) + list(fic_st)
            full_ord_v = [np.nan] * len(y_train) + list(ord_v)

            plot_data[prod][m_name] = full_preds
            plot_data[prod][f'{m_name}_FactStock'] = full_f_st
            plot_data[prod][f'{m_name}_FictionalStock'] = full_fic_st
            plot_data[prod][f'{m_name}_OrderVolume'] = full_ord_v

            if 'Interval' in m_name:
                plot_data[prod]['Interval'] = full_preds
            elif m_name == 'GB':
                plot_data[prod]['GB'] = full_preds
            elif m_name == 'Ridge':
                plot_data[prod]['Ridge'] = full_preds
            
    return pd.DataFrame(summary_results), plot_data

def get_abc_xyz_analysis(df, prices_df):
    # Revenue for each product
    analysis = df.groupby('Product')['Demand'].sum().reset_index()
    
    # Merge with prices and counting total profit
    analysis = analysis.merge(prices_df[['Product', 'Sell_Price', 'Buy_Price']], on='Product')
    analysis['Total_Profit'] = analysis['Demand'] * (analysis['Sell_Price'] - analysis['Buy_Price'])
    
    # ABC analysis
    analysis = analysis.sort_values('Total_Profit', ascending=False)
    analysis['CumSum'] = analysis['Total_Profit'].cumsum()
    total_profit = analysis['Total_Profit'].sum()
    analysis['CumPercent'] = analysis['CumSum'] / total_profit * 100
    
    def abc_classify(percent):
        if percent <= 80: return 'A'
        if percent <= 95: return 'B'
        return 'C'
    
    analysis['ABC'] = analysis['CumPercent'].apply(abc_classify)
    
    # XYZ analysis
    # Counting variation
    variation = df.groupby('Product')['Demand'].agg(['std', 'mean']).reset_index()
    variation['CV'] = variation['std'] / variation['mean']
    
    def xyz_classify(cv):
        if cv <= 0.1: return 'X'
        if cv <= 0.25: return 'Y'
        return 'Z'
    
    variation['XYZ'] = variation['CV'].apply(xyz_classify)
    
    res = analysis.merge(variation[['Product', 'XYZ', 'CV']], on='Product')
    return res[['Product', 'ABC', 'XYZ', 'CV']].rename(columns={'Product': 'Продукт', 'CV': 'Коэф. вариации'})

# UI
st.title("Прогноз спроса и симуляция стратегии управления")

with st.sidebar:
    data_file = st.file_uploader("Загрузить CSV", type="csv")
    price_file = st.file_uploader("Загрузить CSV с ценами (опционально)", type="csv")
    test_val = st.slider("Тестовая выборка (доля)", 0.1, 0.8, 0.2)
    depth_val = st.select_slider("Глубина Optuna", ["Быстрое", "Обычное", "Долгое"], "Быстрое")

if data_file:
    df = pd.read_csv(data_file, sep=';')
    # print(df)
    # print(df.columns)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%m/%Y')
    #st.write(df)
    st.write("### Настройка параметров материалов")
    
    # Prices loading
    if price_file:
        input_prices_df = pd.read_csv(price_file).reset_index(drop=True)
        
        defaults = {
            "Buy_Price": 100.0,
            "Sell_Price": 180.0,
            "Storage_Price": 5.0,
            "Utilization_Cost": 0.0,
            "Preservation_A": 1.0,
            "Delivery_Delay": 0
        }
        for col, default_val in defaults.items():
            if col not in input_prices_df.columns:
                input_prices_df[col] = default_val

        if 'Product' in input_prices_df.columns:
            input_prices_df['Product'] = input_prices_df['Product'].astype(str).str.strip()

        missing_prods = [p for p in df['Product'].unique() if p not in input_prices_df['Product'].values]
        if missing_prods:
            extra_df = pd.DataFrame({
                "Product": missing_prods,
                "Buy_Price": [100.0]*len(missing_prods),
                "Sell_Price": [180.0]*len(missing_prods),
                "Storage_Price": [5.0]*len(missing_prods),
                "Utilization_Cost": [0.0]*len(missing_prods),
                "Preservation_A": [1.0]*len(missing_prods),
                "Delivery_Delay": [0]*len(missing_prods)
            })
            input_prices_df = pd.concat([input_prices_df, extra_df], ignore_index=True)
    else:
        unique_products = df['Product'].unique()
        input_prices_df = pd.DataFrame({
            "Product": unique_products,
            "Buy_Price": [100.0]*len(unique_products),
            "Sell_Price": [180.0]*len(unique_products),
            "Storage_Price": [5.0]*len(unique_products),
            "Utilization_Cost": [0.0]*len(unique_products),
            "Preservation_A": [1.0]*len(unique_products),
            "Delivery_Delay": [0]*len(unique_products)
        })

    rename_map = {
        "Product": "Продукт",
        "Buy_Price": "Цена закупки (руб.)", 
        "Sell_Price": "Цена продажи (руб.)",
        "Storage_Price": "Цена хранения (руб.)", 
        "Utilization_Cost": "Стоимость утилизации (руб.)", 
        "Preservation_A": "Сохранность (A)",
        "Delivery_Delay": "Задержка поставки (мес.)"
    }
    input_prices_df = input_prices_df.rename(columns=rename_map)
    
    prices_df = st.data_editor(input_prices_df.rename(columns=rename_map), key="editor", hide_index=True)
    
    reverse_map = {v: k for k, v in rename_map.items()}
    prices_df_ready = prices_df.rename(columns=reverse_map)
    
    if st.button("🚀 Запустить расчет и графики"):
        df_sorted = df.sort_values('Date')
        unique_dates = df_sorted['Date'].unique()
        split_point = unique_dates[int(len(unique_dates) * (1 - test_val))]
        
        train_df = df_sorted[df_sorted['Date'] < split_point]
        res_df, plots = run_full_analysis_with_plots(df, prices_df_ready, test_val, depth_val)

        st.session_state['train_df'] = train_df
        st.session_state['res_df'] = res_df
        st.session_state['plots'] = plots

    if 'res_df' in st.session_state:
        st.write("---")
        res_df = st.session_state['res_df']
        plots = st.session_state['plots']
        train_df = st.session_state['train_df']

        # ABC/XYZ
        st.write("### ABC/XYZ Классификация")
        abc_xyz_df = get_abc_xyz_analysis(train_df, prices_df_ready) 
        
        col_m1, col_m2 = st.columns([1, 2])
        with col_m1:
            matrix_stats = abc_xyz_df.groupby(['ABC', 'XYZ']).size().unstack(fill_value=0)
            st.write("**Матрица (кол-во товаров):**")
            st.dataframe(matrix_stats)
        with col_m2:
            st.write("**Справка по категориям:**")
            st.info("""

            * **A:** Высокая важность, 80% дохода, 20% ассортимента

            * **B:** Средняя важность, 15% дохода, 30% ассортимента.

            * **C:** Низкая важность, 5% дохода, 50% ассортимента.

            * **X:** Стабильный спрос, высокая точность прогноза (коэф. вариации 0-10%).

            * **Y:** Колебания спроса, средняя точность прогноза (коэф. вариации 10-25%).

            * **Z:** Нестабильный спрос, низкая точность прогноза (коэф. вариации более 25%).

            """)
        
        with st.expander("Посмотреть полный список категорий"):

            st.dataframe(abc_xyz_df, hide_index=True, use_container_width=True)

        st.write("---")
        st.write("### Лучшие стратегии по продуктам")
        best_per_prod = res_df.sort_values('Прибыль (руб.)', ascending=False).drop_duplicates('Продукт')
        best_with_abc = best_per_prod.merge(abc_xyz_df[['Продукт', 'ABC', 'XYZ']], on='Продукт')
        st.dataframe(best_with_abc, hide_index=True)
        
        # All models params
        with st.expander("Посмотреть все обученные модели и их настройки"):
            st.dataframe(res_df, use_container_width=True, hide_index=True)

        st.write("---")
        st.write("### Детальные графики и Логистика запасов")
        
        selected_p = st.selectbox("Выберите материал:", list(plots.keys()))
        p_df = plots[selected_p].sort_values('Date')
        
        prod_best = best_per_prod[best_per_prod['Продукт'] == selected_p].iloc[0]
        best_strat_key = prod_best['Стратегия']

        st.write(f"По умолчанию выбрана экономически оптимальная стратегия: **{best_strat_key}**")
        chosen_strat = st.radio("Показать на графике метрики цепи поставок для:", ['GB', 'Ridge', 'Интервальная (max)'], index=['GB', 'Ridge', 'Interval' in best_strat_key].index(True) if 'Interval' in best_strat_key else ['GB', 'Ridge', 'Интервальная (max)'].index(best_strat_key))
        
        strat_column_map = {'GB': 'GB', 'Ridge': 'Ridge', 'Интервальная (max)': [c for c in p_df.columns if 'Interval' in c and 'Stock' not in c and 'Order' not in c][0]}
        active_strat = strat_column_map[chosen_strat]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 2, 4]})
        split_date = p_df[p_df['Is_Test'] == True]['Date'].min()
        
        ax1.plot(p_df['Date'], p_df['Actual'], label='Реальный спрос (Факт)', color='black', marker='o', alpha=0.4)
        ax1.plot(p_df['Date'], p_df[active_strat], '--', label=f'Целевой уровень запаса ({chosen_strat})', linewidth=2, color='darkorange')
        ax1.axvline(x=split_date, color='blue', linestyle=':', alpha=0.7, label='Разделение Train/Test')
        ax1.axvspan(p_df['Date'].min(), split_date, color='gray', alpha=0.05, label='Зона обучения')
        ax1.set_title("Анализ спроса и целевого уровня пополнения")
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.2)

        test_only = p_df[p_df['Is_Test'] == True]
        ax2.plot(test_only['Date'], test_only[f'{active_strat}_FactStock'], label='Фактический запас на складе', color='green', marker='s', linewidth=2)
        ax2.plot(test_only['Date'], test_only['Actual'], label='Фактический спрос', color='black', linestyle='--', marker='o', alpha=0.6)
        ax2.plot(test_only['Date'], test_only[f'{active_strat}_FictionalStock'], label='Фиктивный запас (Склад + В пути)', color='teal', linestyle=':', alpha=0.8)
        ax2.bar(test_only['Date'], test_only[f'{active_strat}_OrderVolume'], width=15, label='Объем нового заказа (Закупка)', color='crimson', alpha=0.6)
        ax2.set_title("Динамика запасов и поставок (Тестовый период)")
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.2)
        
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        ax3.plot(p_df['Date'], p_df['Actual'], label='Реальный спрос', color='black', marker='o', alpha=0.3, zorder=1)
        
        ax3.plot(test_only['Date'], test_only['Actual'], color='black', marker='o', linewidth=2, label='Тест (факт)', zorder=2)

        ax3.plot(p_df['Date'], p_df['GB'], '--', label='Прогноз GB', linewidth=2)
        ax3.plot(p_df['Date'], p_df['Ridge'], '--', label='Прогноз Ridge', linewidth=2)
        ax3.step(p_df['Date'], p_df['Interval'], label='Максимальный спрос за интервал', where='post', color='red', alpha=0.7)

        ax3.axvline(x=split_date, color='blue', linestyle='-', alpha=0.5, label='Разделение Train/Test')
        
        ax3.axvspan(p_df['Date'].min(), split_date, color='gray', alpha=0.1, label='Зона обучения')
        
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax3.grid(True, alpha=0.2)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.write(f"#### Ведомость движения запасов и заказов ({chosen_strat})")
        
        forecast_table = test_only.copy()
        forecast_table['Date'] = forecast_table['Date'].dt.strftime('%Y-%m')
        
        cols_to_display = [
            'Date', 'Actual', active_strat, 
            f'{active_strat}_FactStock', f'{active_strat}_FictionalStock', f'{active_strat}_OrderVolume'
        ]
        
        forecast_table = forecast_table[cols_to_display]
        forecast_table.columns = [
            'Дата', 'Фактический спрос', 'Целевой уровень (План)', 
            'Запас на начало мес. (Склад)', 'Фиктивный запас (Склад+Путь)', 'Сделанный заказ (Закупка)'
        ]
        
        st.dataframe(
            forecast_table.style.format(precision=2).background_gradient(subset=['Сделанный заказ (Закупка)'], cmap='Reds'), 
            use_container_width=True, hide_index=True
        )
        
        st.success(f"Оптимальная стратегия по экономическому критерию: **{prod_best['Стратегия']}** | Ожидаемая прибыль: {round(prod_best['Прибыль (руб.)'], 2)} руб.")