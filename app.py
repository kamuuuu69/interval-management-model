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
            model = Ridge(alpha=trial.suggest_float("alpha", 0.1, 10.0))
        
        scores = []
        for train_idx, val_idx in tscv.split(X):
            model.fit(X[train_idx], y[train_idx])
            scores.append(np.sqrt(mean_squared_error(y[val_idx], model.predict(X[val_idx]))))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

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
        prod_df['Demand_Lag1'] = prod_df['Demand'].shift(1)
        prod_df = prod_df.dropna(subset=['Demand_Lag1'])
        
        price_row = prices_df[prices_df['Product'] == prod]
        if price_row.empty or len(prod_df) < 10: continue
        p = price_row.iloc[0]
        
        # Preservation
        A = p.get('Preservation_A', 1.0)
        
        X = prod_df[['Demand_Lag1']].values
        y = prod_df['Demand'].values
        dates = prod_df['Date']
        
        split_idx = int(len(prod_df) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        test_dates = dates[split_idx:]
        
        # Training
        best_params_gb = optimize_model(X_train, y_train, "GB", depth)
        best_params_ridge = optimize_model(X_train, y_train, "Ridge", depth)

        

        model_gb = GradientBoostingRegressor(**best_params_gb).fit(X_train, y_train)
        model_ridge = Ridge(**best_params_ridge).fit(X_train, y_train)

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
        def get_metrics(strategy_name):
        
            current_stock = prod_df['Stock'].iloc[split_idx]
            prof, short = 0, 0
            
            # Window initialization
            history = list(y_train[-best_w:]) 
            
            # Dynamic window predictions
            dynamic_preds = []
            
            for i in range(len(y_test)):
                actual_d = y_test[i]
                
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
                # Order to target level
                u = max(0, target_level - current_stock)
                available = current_stock + u
                sold = min(actual_d, available)
                
                # Preservation
                # Remains after sales
                remains_after_sales = max(0, available - actual_d)
                
                # Spoiled units
                spoiled_units = remains_after_sales * (1 - A)
                
                # Profit and shortage
                revenue = sold * p['Sell_Price']
                buy_cost = u * p['Buy_Price']
                storage_cost = remains_after_sales * p['Storage_Price']
                spoiled_cost = spoiled_units * p['Utilization_Cost']
                
                prof += (revenue - buy_cost - storage_cost - spoiled_cost)
                short += max(0, actual_d - available)

                # Stock update
                current_stock = remains_after_sales * A
                
                # Window update
                history.append(actual_d)
                history.pop(0)

            # RMSE on test
            rmse_test = np.sqrt(mean_squared_error(y_test, dynamic_preds))
            
            return prof, short, dynamic_preds, rmse_test

        # Results visualization
        plot_data[prod] = pd.DataFrame({
            'Date': dates, 
            'Actual': y,
            'Is_Test': [False] * len(y_train) + [True] * len(y_test)
        })

        for m_name in ['GB', 'Ridge', f'Interval_{best_w}m']:
            internal_name = m_name.split('_')[0] if 'Interval' in m_name else m_name

            pr, sh, d_preds, rmse = get_metrics(internal_name)
            summary_results.append({
                'Продукт': prod, 
                'Стратегия': m_name, 
                'Прибыль': pr, 
                'Дефицит': sh,
                'RMSE': round(rmse, 2),
                'Параметры': model_configs.get(m_name, "N/A")
            })
            
            full_preds = [np.nan] * len(y_train) + list(d_preds)

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
st.title("Прогноз спроса")

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
    columns_ru = ["Продукт", "Цена закупки", "Цена продажи", "Цена хранения", "Стоимость утилизации", "Сохранность (A)"]
    st.write("### Настройка параметров материалов")
    
    # Prices loading
    if price_file:
        input_prices_df = pd.read_csv(price_file, index_col=0)
        # Missing files have 0s in it's prices
        missing_prods = [p for p in df['Product'].unique() if p not in input_prices_df['Product'].values]
        if missing_prods:
            extra_df = pd.DataFrame({
                "Product": missing_prods,
                "Buy_Price": [0.0]*len(missing_prods),
                "Sell_Price": [0.0]*len(missing_prods),
                "Storage_Price": [0.0]*len(missing_prods),
                "Utilization_Cost": [0.0]*len(missing_prods),
                "Preservation_A": [1.0]*len(missing_prods)
            })
            input_prices_df = pd.concat([input_prices_df, extra_df], ignore_index=True)
    else:
        # Default prices
        input_prices_df = pd.DataFrame({
            "Product": df['Product'].unique(),
            "Buy_Price": [100.0]*len(df['Product'].unique()),
            "Sell_Price": [180.0]*len(df['Product'].unique()),
            "Storage_Price": [5.0]*len(df['Product'].unique()),
            "Utilization_Cost": [0.0]*len(df['Product'].unique()),
            "Preservation_A": [1.0]*len(df['Product'].unique())
        })
    # Rename prices to russian
    rename_map = {
            "Product": "Продукт", "Buy_Price": "Цена закупки", "Sell_Price": "Цена продажи",
            "Storage_Price": "Цена хранения", "Utilization_Cost": "Стоимость утилизации", "Preservation_A": "Сохранность (A)"
    }
    input_prices_df = input_prices_df.rename(columns=rename_map)
    # Prices editor
    prices_df = st.data_editor(input_prices_df, key="editor", hide_index=True)
    reverse_map = {v: k for k, v in rename_map.items()}
    prices_df_ready = prices_df.rename(columns=reverse_map)
    
    if st.button("🚀 Запустить расчет и графики"):
        res_df, plots = run_full_analysis_with_plots(df, prices_df_ready, test_val, depth_val)
        
        st.session_state['res_df'] = res_df
        st.session_state['plots'] = plots

    if 'res_df' in st.session_state:

        st.write("---")
        res_df = st.session_state['res_df']
        plots = st.session_state['plots']

        # ABC/XYZ analysis
        st.write("### ABC/XYZ Классификация")
        
        abc_xyz_df = get_abc_xyz_analysis(df, prices_df_ready) 
        
        col_m1, col_m2 = st.columns([1, 2])
        with col_m1:
            # Матрица распределения
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
        # ----------------------------------

        st.write("---")
        st.write("### 🏆 Лучшие стратегии по продуктам")
        
        # Resilts merging
        best_per_prod = res_df.sort_values('Прибыль', ascending=False).drop_duplicates('Продукт')
        best_with_abc = best_per_prod.merge(abc_xyz_df[['Продукт', 'ABC', 'XYZ']], left_on='Продукт', right_on='Продукт')
        
        st.dataframe(best_with_abc, hide_index=True)
        
        # All models params
        with st.expander("🔍 Посмотреть все обученные модели и их настройки"):
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            
        # All models on every product type
        with st.expander("Показать все результаты"):
            st.dataframe(res_df)
        
        st.write("---")
        st.write("### 📊 Детальные графики")
        
        selected_p = st.selectbox("Выберите материал:", list(plots.keys()))
        p_df = plots[selected_p].sort_values('Date')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        split_date = p_df[p_df['Is_Test'] == True]['Date'].min()
        
        ax.plot(p_df['Date'], p_df['Actual'], label='Реальный спрос', color='black', marker='o', alpha=0.3, zorder=1)
        
        test_only = p_df[p_df['Is_Test'] == True]
        ax.plot(test_only['Date'], test_only['Actual'], color='black', marker='o', linewidth=2, label='Тест (факт)', zorder=2)

        ax.plot(p_df['Date'], p_df['GB'], '--', label='Прогноз GB', linewidth=2)
        ax.plot(p_df['Date'], p_df['Ridge'], '--', label='Прогноз Ridge', linewidth=2)
        ax.step(p_df['Date'], p_df['Interval'], label='Максимальный спрос за интервал', where='post', color='red', alpha=0.7)

        ax.axvline(x=split_date, color='blue', linestyle='-', alpha=0.5, label='Разделение Train/Test')
        
        ax.axvspan(p_df['Date'].min(), split_date, color='gray', alpha=0.1, label='Зона обучения')
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.2)
        
        st.pyplot(fig)
        
        # Best model for that product type
        prod_best = best_per_prod[best_per_prod['Продукт'] == selected_p].iloc[0]
        st.success(f"Оптимальный выбор: **{prod_best['Стратегия']}** | Ожидаемая прибыль: {round(prod_best['Прибыль'], 2)}")