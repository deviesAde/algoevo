import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import requests
import io
from urllib.parse import urlparse

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="Genetic Algorithm - Knapsack",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# CUSTOM CSS
# ============================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.4rem;
        color: #2e86ab;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
        font-weight: 600;
    }
    .info-box {
        background: #d1ecf1;
        border: 2px solid #17a2b8;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        color: #0c5460;
    }
    .warning-box {
        background: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        color: #856404;
    }
    .stButton button {
        background: #667eea;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 2rem;
        font-weight: bold;
    }
    .success-box {
        background: #d4edda;
        border: 2px solid #28a745;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# GA PARAMETERS
# ============================
POP_SIZE = 50
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
GENERATIONS = 100
TOURNAMENT_SIZE = 2

# ============================
# GA FUNCTIONS (TIDAK DIUBAH)
# ============================

def fitness(chrom, weights, profits, capacity):
    total_w = np.sum(chrom * weights)
    total_p = np.sum(chrom * profits)

    if total_w <= capacity:
        return total_p
    else:
        return 0

def init_population(n_items):
    return np.random.randint(0, 2, size=(POP_SIZE, n_items))

def tournament_selection(pop, fit_vals):
    i, j = np.random.choice(len(pop), 2, replace=False)
    return pop[i] if fit_vals[i] > fit_vals[j] else pop[j]

def crossover(p1, p2):
    if np.random.rand() < CROSSOVER_RATE:
        point = np.random.randint(1, len(p1))
        return np.concatenate([p1[:point], p2[point:]]), \
               np.concatenate([p2[:point], p1[point:]])
    return p1.copy(), p2.copy()

def mutation(ch):
    for i in range(len(ch)):
        if np.random.rand() < MUTATION_RATE:
            ch[i] = 1 - ch[i]
    return ch

def run_ga(weights, profits, capacity):
    n_items = len(weights)
    pop = init_population(n_items)

    best_fitness_per_gen = []
    avg_fitness_per_gen = []

    best_chrom = None
    best_fit = -9999

    progress_bar = st.progress(0)
    status_text = st.empty()

    for gen in range(GENERATIONS):
        fit_vals = np.array([fitness(ind, weights, profits, capacity) for ind in pop])

        best_fitness_per_gen.append(fit_vals.max())
        avg_fitness_per_gen.append(fit_vals.mean())

        if fit_vals.max() > best_fit:
            best_fit = fit_vals.max()
            best_chrom = pop[np.argmax(fit_vals)].copy()

        new_pop = []

        while len(new_pop) < POP_SIZE:
            p1 = tournament_selection(pop, fit_vals)
            p2 = tournament_selection(pop, fit_vals)

            c1, c2 = crossover(p1, p2)

            new_pop.append(mutation(c1))
            if len(new_pop) < POP_SIZE:
                new_pop.append(mutation(c2))

        pop = np.array(new_pop)
        
        progress = (gen + 1) / GENERATIONS
        progress_bar.progress(progress)
        status_text.text(f"Generation {gen + 1}/{GENERATIONS}")

    progress_bar.empty()
    status_text.empty()
    
    return best_chrom, best_fit, best_fitness_per_gen, avg_fitness_per_gen

# ============================
# DATASET FUNCTIONS
# ============================

def load_sample_dataset():
    """Membuat dataset sample berdasarkan contoh yang diberikan"""
    data = {
        'Weights': ['[46, 40, 42, 38, 10]'],
        'Prices': ['[12, 19, 19, 15, 8]'],
        'Capacity': [40],
        'Best_picks': ['[0, 1, 0, 0, 0]'],
        'Best_price': [19]
    }
    return pd.DataFrame(data)

def parse_array_string(arr_str):
    """Mengubah string array menjadi list of integers"""
    if isinstance(arr_str, str):
        # Remove brackets and split by space or comma
        arr_str = arr_str.replace('[', '').replace(']', '')
        # Split by comma or space
        elements = arr_str.replace(',', ' ').split()
        return [int(x.strip()) for x in elements if x.strip()]
    elif isinstance(arr_str, list):
        return arr_str
    else:
        return []

def safe_convert_to_int(value):
    """Mengkonversi value ke integer dengan aman"""
    try:
        if isinstance(value, (int, float, np.number)):
            return int(value)
        elif isinstance(value, str):
            # Remove any non-numeric characters except minus
            cleaned = ''.join(c for c in value if c.isdigit() or c == '-')
            return int(cleaned) if cleaned else 0
        else:
            return int(value)
    except (ValueError, TypeError):
        return 0

def detect_column_names(df):
    """Mendeteksi nama kolom yang sesuai dalam dataset"""
    column_mapping = {}
    
    # Cari kolom weights
    weight_candidates = ['Weights', 'weights', 'Weight', 'weight', 'w', 'W']
    for col in df.columns:
        for candidate in weight_candidates:
            if candidate.lower() in col.lower():
                column_mapping['weights'] = col
                break
        if 'weights' in column_mapping:
            break
    
    # Cari kolom prices/profits
    price_candidates = ['Prices', 'prices', 'Price', 'price', 'Profits', 'profits', 'Profit', 'profit', 'p', 'P']
    for col in df.columns:
        for candidate in price_candidates:
            if candidate.lower() in col.lower():
                column_mapping['prices'] = col
                break
        if 'prices' in column_mapping:
            break
    
    # Cari kolom capacity
    capacity_candidates = ['Capacity', 'capacity', 'Cap', 'cap', 'C', 'c', 'MaxWeight', 'maxweight']
    for col in df.columns:
        for candidate in capacity_candidates:
            if candidate.lower() in col.lower():
                column_mapping['capacity'] = col
                break
        if 'capacity' in column_mapping:
            break
    
    # Cari kolom best price (opsional)
    best_price_candidates = ['Best_price', 'best_price', 'BestPrice', 'bestprice', 'Optimal', 'optimal', 'Best', 'best']
    for col in df.columns:
        for candidate in best_price_candidates:
            if candidate.lower() in col.lower():
                column_mapping['best_price'] = col
                break
    
    return column_mapping

def validate_and_clean_dataset(df, column_mapping):
    """Validasi dan bersihkan dataset"""
    valid_rows = []
    
    for idx, row in df.iterrows():
        try:
            # Parse weights dan prices
            weights = parse_array_string(row[column_mapping['weights']])
            prices = parse_array_string(row[column_mapping['prices']])
            
            # Konversi capacity ke integer
            capacity = safe_convert_to_int(row[column_mapping['capacity']])
            
            # Validasi data
            if (len(weights) > 0 and len(prices) > 0 and 
                len(weights) == len(prices) and capacity > 0):
                
                # Jika best_price tersedia, konversi juga
                best_price = None
                if 'best_price' in column_mapping:
                    best_price = safe_convert_to_int(row[column_mapping['best_price']])
                
                valid_rows.append({
                    'index': idx,
                    'weights': weights,
                    'prices': prices,
                    'capacity': capacity,
                    'best_price': best_price,
                    'original_row': row
                })
                
        except Exception as e:
            st.warning(f"Baris {idx} dilewati karena error: {e}")
            continue
    
    return valid_rows

def test_algorithm_performance(dataset):
    """Menguji performa algoritma GA pada dataset"""
    results = []
    
    # Deteksi nama kolom
    column_mapping = detect_column_names(dataset)
    
    st.write("**Kolom yang terdeteksi:**", column_mapping)
    
    # Validasi kolom yang diperlukan
    required_columns = ['weights', 'prices', 'capacity']
    missing_columns = [col for col in required_columns if col not in column_mapping]
    
    if missing_columns:
        st.error(f"Kolom yang diperlukan tidak ditemukan: {missing_columns}")
        st.write("Kolom yang tersedia:", list(dataset.columns))
        return pd.DataFrame()
    
    # Validasi dan bersihkan dataset
    st.info("Memvalidasi dataset...")
    valid_data = validate_and_clean_dataset(dataset, column_mapping)
    
    if len(valid_data) == 0:
        st.error("Tidak ada data yang valid setelah validasi.")
        return pd.DataFrame()
    
    st.success(f"Found {len(valid_data)} valid rows out of {len(dataset)} total rows")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, data_item in enumerate(valid_data):
        weights = data_item['weights']
        prices = data_item['prices']
        capacity = data_item['capacity']
        best_price_expected = data_item['best_price']
        
        # Jalankan GA
        best_chrom, best_fit, best_list, avg_list = run_ga(
            np.array(weights), 
            np.array(prices), 
            capacity
        )
        
        # Hitung metrik performa
        total_weight = np.sum(best_chrom * weights)
        
        # Hitung accuracy jika best_price tersedia
        if best_price_expected is not None and best_price_expected > 0:
            accuracy = (best_fit / best_price_expected * 100)
            is_optimal = 1 if best_fit == best_price_expected else 0
        else:
            accuracy = None
            is_optimal = None
        
        capacity_usage = (total_weight / capacity * 100) if capacity > 0 else 0
        
        result_item = {
            'Test_Case': i + 1,
            'Original_Index': data_item['index'],
            'Items_Count': len(weights),
            'GA_Profit': best_fit,
            'Capacity_Used': total_weight,
            'Capacity_Available': capacity,
            'Capacity_Usage_Percentage': capacity_usage,
            'Selected_Items': np.sum(best_chrom)
        }
        
        # Tambahkan best_price jika tersedia
        if best_price_expected is not None:
            result_item['Expected_Profit'] = best_price_expected
            result_item['Accuracy_Percentage'] = accuracy
            result_item['Is_Optimal'] = is_optimal
        
        results.append(result_item)
        
        # Update progress
        progress = (i + 1) / len(valid_data)
        progress_bar.progress(progress)
        status_text.text(f"Testing case {i + 1}/{len(valid_data)} - {len(weights)} items")
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

# ============================
# MAIN UI
# ============================
st.markdown('<div class="main-header">Genetic Algorithm Knapsack Optimizer</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <strong>Tujuan:</strong> Memilih kombinasi barang yang memberikan profit maksimal tanpa melebihi kapasitas knapsack menggunakan Genetic Algorithm.
</div>
""", unsafe_allow_html=True)

# Tab untuk memilih mode
tab1, tab2 = st.tabs(["🔧 Manual Input", "📊 Test Performance"])

with tab1:
    st.markdown('<div class="sub-header">Input Data Barang Manual</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        n = st.number_input("**Jumlah Barang**", 1, 100, 5, help="Masukkan jumlah barang yang tersedia", key="manual_n")

    with col2:
        st.markdown("**Pilih Kapasitas Knapsack:**")
        weight_options = {
            "Kecil (30) - Lebih menantang": 30,
            "Sedang (50) - Optimal": 50,
            "Besar (80) - Lebih longgar": 80,
            "Custom - Tentukan sendiri": "custom"
        }
        
        selected_option = st.selectbox(
            "Pilih kapasitas:",
            list(weight_options.keys()),
            key="manual_capacity"
        )
        
        if weight_options[selected_option] == "custom":
            W = st.number_input("Masukkan kapasitas custom:", 1, 1000, 50, key="manual_custom_capacity")
        else:
            W = weight_options[selected_option]
            st.success(f"Kapasitas knapsack: {W}")

    # Input data barang manual
    st.markdown("### Detail Data Barang")

    weights_manual = []
    profits_manual = []

    if n > 0:
        cols_per_row = 4
        rows = (n + cols_per_row - 1) // cols_per_row

        for row in range(rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                item_idx = row * cols_per_row + col_idx
                if item_idx < n:
                    with cols[col_idx]:
                        st.markdown(f"**Barang {item_idx + 1}**")
                        w = st.number_input(f"Weight", min_value=1, value=np.random.randint(5, 20), key=f"manual_w{item_idx}")
                        p = st.number_input(f"Profit", min_value=1, value=np.random.randint(15, 40), key=f"manual_p{item_idx}")
                        weights_manual.append(w)
                        profits_manual.append(p)

    # Run GA untuk manual input
    if len(weights_manual) > 0 and len(profits_manual) > 0:
        st.markdown("---")
        
        st.markdown("### Konfigurasi Saat Ini")
        config_col1, config_col2, config_col3 = st.columns(3)
        
        with config_col1:
            st.markdown("**Data Barang:**")
            st.write(f"- Jumlah: {len(weights_manual)} barang")
            st.write(f"- Total weight: {sum(weights_manual)}")
            st.write(f"- Total profit: Rp {sum(profits_manual):,}")
        
        with config_col2:
            st.markdown("**Knapsack:**")
            st.write(f"- Kapasitas: {W}")
            st.write(f"- Tingkat penggunaan: {sum(weights_manual)/W*100:.1f}%")
        
        with config_col3:
            st.markdown("**Genetic Algorithm:**")
            st.write(f"- Populasi: {POP_SIZE}")
            st.write(f"- Generasi: {GENERATIONS}")
            st.write(f"- Crossover: {CROSSOVER_RATE}")
            st.write(f"- Mutasi: {MUTATION_RATE}")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("**Jalankan Genetic Algorithm**", type="primary", use_container_width=True, key="manual_run"):
                weights_array = np.array(weights_manual)
                profits_array = np.array(profits_manual)
                
                with st.spinner("Menjalankan Genetic Algorithm..."):
                    best_chrom, best_fit, best_list, avg_list = run_ga(weights_array, profits_array, W)

                display_results(best_chrom, best_fit, best_list, avg_list, weights_array, profits_array, W)

with tab2:
    st.markdown('<div class="sub-header">Test Performance dengan Dataset</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>Dataset Knapsack Problem:</strong> Menguji performa algoritma GA pada berbagai kasus knapsack 
        dengan solusi optimal yang sudah diketahui. Dataset bisa diupload atau menggunakan sample.
    </div>
    """, unsafe_allow_html=True)
    
    # Pilihan dataset
    dataset_option = st.radio(
        "Pilih sumber dataset:",
        ["📁 Upload CSV", "📊 Gunakan Sample Dataset"],
        horizontal=True
    )
    
    dataset = None
    
    if dataset_option == "📁 Upload CSV":
        uploaded_file = st.file_uploader("Upload file CSV dataset", type=['csv'])
        
        if uploaded_file is not None:
            try:
                dataset = pd.read_csv(uploaded_file)
                st.success(f"Dataset berhasil diupload! {len(dataset)} baris data.")
                
                # Tampilkan kolom yang terdeteksi
                column_mapping = detect_column_names(dataset)
                st.markdown("**Kolom yang terdeteksi:**")
                st.json(column_mapping)
                
                # Tampilkan preview
                st.markdown("**Preview Dataset:**")
                st.dataframe(dataset.head(), use_container_width=True)
                
                # Tampilkan info tipe data
                st.markdown("**Info Tipe Data:**")
                st.write(dataset.dtypes)
                
            except Exception as e:
                st.error(f"Error membaca file: {e}")
    
    else:  # Sample dataset
        dataset = load_sample_dataset()
        st.markdown("""
        <div class="warning-box">
            <strong>Sample Dataset:</strong> Menggunakan dataset contoh dengan format: Weights, Prices, Capacity, Best_picks, Best_price
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Sample Dataset:**")
        st.dataframe(dataset, use_container_width=True)
    
    # Tombol test performance
    if dataset is not None and len(dataset) > 0:
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🧪 Test Performance Algorithm", type="primary", use_container_width=True):
                with st.spinner("Menguji performa algoritma pada dataset..."):
                    results_df = test_algorithm_performance(dataset)
                
                if len(results_df) == 0:
                    st.error("Tidak ada hasil yang dapat diproses. Periksa format dataset.")
                else:
                    # Tampilkan hasil performance
                    st.markdown('<div class="sub-header">Hasil Test Performance</div>', unsafe_allow_html=True)
                    
                    # Summary statistics
                    total_cases = len(results_df)
                    
                    # Check if expected profit column exists
                    has_expected_profit = 'Expected_Profit' in results_df.columns
                    
                    if has_expected_profit:
                        optimal_cases = results_df['Is_Optimal'].sum()
                        accuracy_avg = results_df['Accuracy_Percentage'].mean()
                    
                    capacity_usage_avg = results_df['Capacity_Usage_Percentage'].mean()
                    avg_profit = results_df['GA_Profit'].mean()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Test Cases", total_cases)
                    with col2:
                        if has_expected_profit:
                            st.metric("Optimal Solutions", f"{optimal_cases}/{total_cases}")
                        else:
                            st.metric("Average Profit", f"{avg_profit:.1f}")
                    with col3:
                        if has_expected_profit:
                            st.metric("Average Accuracy", f"{accuracy_avg:.1f}%")
                        else:
                            st.metric("Avg Items Selected", f"{results_df['Selected_Items'].mean():.1f}")
                    with col4:
                        st.metric("Avg Capacity Usage", f"{capacity_usage_avg:.1f}%")
                    
                    # Tampilkan detail results
                    st.markdown("#### Detail Hasil Test")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Visualisasi hasil
                    st.markdown("#### Visualisasi Performa")
                    
                    fig1 = go.Figure()
                    
                    if has_expected_profit:
                        fig1.add_trace(go.Scatter(
                            x=results_df['Test_Case'],
                            y=results_df['Accuracy_Percentage'],
                            mode='lines+markers',
                            name='Accuracy (%)',
                            line=dict(color='#28a745', width=3)
                        ))
                        fig1.update_layout(
                            title="Accuracy per Test Case",
                            xaxis_title="Test Case",
                            yaxis_title="Accuracy (%)",
                            height=400
                        )
                    else:
                        fig1.add_trace(go.Scatter(
                            x=results_df['Test_Case'],
                            y=results_df['GA_Profit'],
                            mode='lines+markers',
                            name='GA Profit',
                            line=dict(color='#1f77b4', width=3)
                        ))
                        fig1.update_layout(
                            title="GA Profit per Test Case",
                            xaxis_title="Test Case",
                            yaxis_title="Profit",
                            height=400
                        )
                    
                    fig2 = go.Figure()
                    if has_expected_profit:
                        fig2.add_trace(go.Bar(
                            x=results_df['Test_Case'],
                            y=results_df['GA_Profit'],
                            name='GA Profit',
                            marker_color='#1f77b4'
                        ))
                        fig2.add_trace(go.Scatter(
                            x=results_df['Test_Case'],
                            y=results_df['Expected_Profit'],
                            mode='markers',
                            name='Expected Profit',
                            marker=dict(color='#ff7f0e', size=8, symbol='diamond')
                        ))
                        fig2.update_layout(
                            title="GA Profit vs Expected Profit",
                            xaxis_title="Test Case",
                            yaxis_title="Profit",
                            height=400
                        )
                    else:
                        fig2.add_trace(go.Bar(
                            x=results_df['Test_Case'],
                            y=results_df['GA_Profit'],
                            name='GA Profit',
                            marker_color='#1f77b4'
                        ))
                        fig2.update_layout(
                            title="GA Profit per Test Case",
                            xaxis_title="Test Case",
                            yaxis_title="Profit",
                            height=400
                        )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig1, use_container_width=True)
                    with col2:
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Hasil Test",
                        data=csv,
                        file_name="ga_performance_results.csv",
                        mime="text/csv"
                    )

# ============================
# FUNCTION TO DISPLAY RESULTS
# ============================
def display_results(best_chrom, best_fit, best_list, avg_list, weights, profits, capacity):
    st.markdown('<div class="sub-header">Hasil Optimasi</div>', unsafe_allow_html=True)
    
    total_weight = np.sum(best_chrom * weights)
    total_profit = np.sum(best_chrom * profits)
    selected_items = np.sum(best_chrom)
    efficiency = (total_weight / capacity) * 100 if capacity > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Profit", f"Rp {total_profit:,}")
    with col2:
        st.metric("Total Weight", f"{total_weight:,} / {capacity}")
    with col3:
        st.metric("Barang Terpilih", f"{selected_items} item")
    with col4:
        st.metric("Efisiensi", f"{efficiency:.1f}%")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Solusi Terbaik")
        
        st.markdown("**Kromosom Terbaik:**")
        chrom_html = "".join([f"<span style='display:inline-block; padding:8px 12px; margin:3px; background:{'#28a745' if bit==1 else '#dc3545'}; color:white; border-radius:8px; font-weight:bold; font-size:14px;'>{bit}</span>" for bit in best_chrom])
        st.markdown(chrom_html, unsafe_allow_html=True)
        
        st.markdown("#### Barang yang Dipilih")
        selected_indices = np.where(best_chrom == 1)[0]
        if len(selected_indices) > 0:
            selected_data = []
            total_selected_weight = 0
            total_selected_profit = 0
            
            for idx in selected_indices:
                selected_data.append({
                    "Barang": idx + 1,
                    "Weight": weights[idx],
                    "Profit": f"Rp {profits[idx]:,}"
                })
                total_selected_weight += weights[idx]
                total_selected_profit += profits[idx]
            
            selected_df = pd.DataFrame(selected_data)
            st.dataframe(selected_df, use_container_width=True)
            
            st.markdown(f"**Total:** {len(selected_data)} barang | Weight: {total_selected_weight} | Profit: Rp {total_selected_profit:,}")
        else:
            st.warning("Tidak ada barang yang dipilih")
    
    with col2:
        st.markdown("#### Grafik Konvergensi")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(best_list))),
            y=best_list,
            mode='lines',
            name='Fitness Terbaik',
            line=dict(color='#1f77b4', width=4)
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(avg_list))),
            y=avg_list,
            mode='lines',
            name='Fitness Rata-rata',
            line=dict(color='#ff7f0e', width=3, dash='dash')
        ))
        
        fig.update_layout(
            title="Perkembangan Fitness per Generasi",
            xaxis_title="Generasi",
            yaxis_title="Fitness (Profit)",
            hovermode='x unified',
            height=450,
            yaxis=dict(
                range=[0, max(best_list + avg_list) * 1.1] if len(best_list) > 0 else [0, 100]
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Ringkasan Performa")
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.markdown("**Parameter GA:**")
        st.write(f"- Populasi: {POP_SIZE}")
        st.write(f"- Crossover: {CROSSOVER_RATE}")
        st.write(f"- Mutasi: {MUTATION_RATE}")
        st.write(f"- Generasi: {GENERATIONS}")
    
    with summary_col2:
        st.markdown("**Hasil:**")
        st.write(f"- Fitness Terbaik: {best_fit:.2f}")
        st.write(f"- Kapasitas Terpakai: {total_weight}/{capacity}")
        st.write(f"- Barang Terpilih: {selected_items}/{len(weights)}")
        st.write(f"- Efisiensi: {efficiency:.1f}%")
    
    with summary_col3:
        st.markdown("**Status:**")
        if total_weight <= capacity:
            st.success("**SOLUSI VALID**")
            st.write("Tidak melebihi kapasitas")
        else:
            st.error("**MELEBIHI KAPASITAS**")
            st.write("Dikenai penalty")