import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="Genetic Algorithm - Knapsack",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# CUSTOM CSS - IMPROVED
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        border: none;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        color: #155724;
    }
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 2px solid #17a2b8;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        color: #0c5460;
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 2px solid #ffc107;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        color: #856404;
    }
    /* Improved metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 15px !important;
        padding: 1.5rem !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    [data-testid="metric-container"] label {
        color: white !important;
        font-weight: bold !important;
        font-size: 1rem !important;
    }
    [data-testid="metric-container"] div {
        color: white !important;
        font-size: 1.8rem !important;
        font-weight: bold !important;
    }
    /* Sidebar improvements */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
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
# GA FUNCTIONS
# ============================

def fitness(chrom, weights, profits, capacity):
    total_w = np.sum(chrom * weights)
    total_p = np.sum(chrom * profits)

    if total_w <= capacity:
        return total_p  # ∑ profit jika valid
    else:
        return 0  # 0 jika melanggar batas

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

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_placeholder = st.empty()

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
        
        # Update progress
        progress = (gen + 1) / GENERATIONS
        progress_bar.progress(progress)
        status_text.text(f"Generation {gen + 1}/{GENERATIONS}")
        status_placeholder.text(f"Best Fitness: {best_fit:.2f}")

    progress_bar.empty()
    status_text.empty()
    status_placeholder.empty()
    
    return best_chrom, best_fit, best_fitness_per_gen, avg_fitness_per_gen

# ============================
# SIDEBAR - FIXED (sesuai proposal)
# ============================
with st.sidebar:
    st.markdown("## ⚙️ Parameter GA")
    st.markdown("---")
    
    st.markdown("### Parameter Algoritma")
    
    # Sesuai proposal - parameter TIDAK bisa diadjust
    st.markdown("**Ukuran Populasi:** 50")
    st.markdown("**Probabilitas Crossover:** 0.8")
    st.markdown("**Probabilitas Mutasi:** 0.1")
    st.markdown("**Jumlah Generasi:** 100")
    
    st.markdown("---")
    st.markdown("### 📋 Informasi Proposal")
    st.info("""
    **Parameter sesuai proposal:**
    - Populasi: 50 individu
    - Crossover: 0.8 (80%)
    - Mutasi: 0.1 (10%)
    - Generasi: 100
    - Seleksi: Tournament (size=2)
    """)

# ============================
# MAIN UI
# ============================
st.markdown('<div class="main-header">🧬 Genetic Algorithm Knapsack Optimizer</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <strong>🎯 Tujuan:</strong> Memilih kombinasi barang yang memberikan profit maksimal tanpa melebihi kapasitas knapsack menggunakan Genetic Algorithm.
</div>
""", unsafe_allow_html=True)


st.markdown('<div class="sub-header">📥 Input Data Barang</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    n = st.number_input("**Jumlah Barang**", 1, 100, 5, help="Masukkan jumlah barang yang tersedia")

with col2:
    # PILIHAN WEIGHT KNAPSACK
    st.markdown("**🔧 Pilih Kapasitas Knapsack:**")
    weight_options = {
        "Kecil (30) - Lebih menantang": 30,
        "Sedang (50) - Optimal": 50,
        "Besar (80) - Lebih longgar": 80,
        "Custom - Tentukan sendiri": "custom"
    }
    
    selected_option = st.selectbox(
        "Pilih kapasitas:",
        list(weight_options.keys())
    )
    
    if weight_options[selected_option] == "custom":
        W = st.number_input("Masukkan kapasitas custom:", 1, 1000, 50)
    else:
        W = weight_options[selected_option]
        st.success(f"✅ Kapasitas knapsack: {W}")

# Input data barang
st.markdown("### 📋 Detail Data Barang")

# Create input grid
cols_per_row = 4
rows = (n + cols_per_row - 1) // cols_per_row

weights = []
profits = []

for row in range(rows):
    cols = st.columns(cols_per_row)
    for col_idx in range(cols_per_row):
        item_idx = row * cols_per_row + col_idx
        if item_idx < n:
            with cols[col_idx]:
                st.markdown(f"**Barang {item_idx + 1}**")
                w = st.number_input(f"Weight", min_value=1, value=np.random.randint(5, 20), key=f"w{item_idx}")
                p = st.number_input(f"Profit", min_value=1, value=np.random.randint(15, 40), key=f"p{item_idx}")
                weights.append(w)
                profits.append(p)

# ===========================================
# RUN BUTTON & RESULTS
# ===========================================
if ('weights' in locals() and 'profits' in locals() and len(weights) > 0):
    st.markdown("---")
    
    # Display current configuration
    st.markdown("### ⚡ Konfigurasi Saat Ini")
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        st.markdown("**📦 Data Barang:**")
        st.write(f"- Jumlah: {len(weights)} barang")
        st.write(f"- Total weight: {sum(weights)}")
        st.write(f"- Total profit: ${sum(profits)}")
    
    with config_col2:
        st.markdown("**🎒 Knapsack:**")
        st.write(f"- Kapasitas: {W}")
        st.write(f"- Tingkat penggunaan: {sum(weights)/W*100:.1f}%")
    
    with config_col3:
        st.markdown("**🧬 Genetic Algorithm:**")
        st.write(f"- Populasi: {POP_SIZE}")
        st.write(f"- Generasi: {GENERATIONS}")
        st.write(f"- Crossover: {CROSSOVER_RATE}")
        st.write(f"- Mutasi: {MUTATION_RATE}")
    
    # Run button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 **Jalankan Genetic Algorithm**", type="primary", use_container_width=True):
            weights = np.array(weights)
            profits = np.array(profits)
            
            with st.spinner("🔄 Menjalankan Genetic Algorithm..."):
                best_chrom, best_fit, best_list, avg_list = run_ga(weights, profits, W)

            # ===========================================
            # RESULTS SECTION
            # ===========================================
            st.markdown('<div class="sub-header">📊 Hasil Optimasi</div>', unsafe_allow_html=True)
            
            # Metrics
            total_weight = np.sum(best_chrom * weights)
            total_profit = np.sum(best_chrom * profits)
            selected_items = np.sum(best_chrom)
            efficiency = (total_weight / W) * 100 if W > 0 else 0
            
            # Improved metrics display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("💰 Total Profit", f"${total_profit:,.0f}")
            with col2:
                st.metric("⚖️ Total Weight", f"{total_weight:,.0f} / {W}")
            with col3:
                st.metric("📦 Barang Terpilih", f"{selected_items} item")
            with col4:
                st.metric("🎯 Efisiensi", f"{efficiency:.1f}%")
            
            # Results details in two columns
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### 🎯 Solusi Terbaik")
                
                # Chromosome visualization with better styling
                st.markdown("**Kromosom Terbaik:**")
                chrom_html = "".join([f"<span style='display:inline-block; padding:8px 12px; margin:3px; background:{'#28a745' if bit==1 else '#dc3545'}; color:white; border-radius:8px; font-weight:bold; font-size:14px;'>{bit}</span>" for bit in best_chrom])
                st.markdown(chrom_html, unsafe_allow_html=True)
                
                # Selected items table
                st.markdown("#### 📋 Barang yang Dipilih")
                selected_indices = np.where(best_chrom == 1)[0]
                if len(selected_indices) > 0:
                    selected_data = []
                    total_selected_weight = 0
                    total_selected_profit = 0
                    
                    for idx in selected_indices:
                        selected_data.append({
                            "Barang": idx + 1,
                            "Weight": weights[idx],
                            "Profit": f"${profits[idx]}"
                        })
                        total_selected_weight += weights[idx]
                        total_selected_profit += profits[idx]
                    
                    selected_df = pd.DataFrame(selected_data)
                    st.dataframe(selected_df, use_container_width=True)
                    
                    # Summary of selected items
                    st.markdown(f"**Total:** {len(selected_data)} barang | Weight: {total_selected_weight} | Profit: ${total_selected_profit}")
                else:
                    st.warning("❌ Tidak ada barang yang dipilih")
            
            with col2:
                st.markdown("#### 📈 Grafik Konvergensi")
                
                # Create interactive plot with Plotly
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
                    yaxis_title="Fitness",
                    hovermode='x unified',
                    height=450,
                    plot_bgcolor='rgba(240,240,240,0.8)',
                    paper_bgcolor='rgba(240,240,240,0.1)',
                    font=dict(size=12)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Convergence info
                improvement = ((best_list[-1] - best_list[0]) / best_list[0]) * 100 if best_list[0] != 0 else 0
                st.info(f"**📈 Improvement:** {improvement:+.1f}% dari generasi awal")

            # Performance summary
            st.markdown("#### 📋 Ringkasan Performa")
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                st.markdown("**⚙️ Parameter GA:**")
                st.write(f"- Populasi: {POP_SIZE}")
                st.write(f"- Crossover: {CROSSOVER_RATE}")
                st.write(f"- Mutasi: {MUTATION_RATE}")
                st.write(f"- Generasi: {GENERATIONS}")
            
            with summary_col2:
                st.markdown("**📊 Hasil:**")
                st.write(f"- Fitness Terbaik: {best_fit:.2f}")
                st.write(f"- Kapasitas Terpakai: {total_weight}/{W}")
                st.write(f"- Barang Terpilih: {selected_items}/{len(weights)}")
                st.write(f"- Efisiensi: {efficiency:.1f}%")
            
            with summary_col3:
                st.markdown("**✅ Status:**")
                if total_weight <= W:
                    st.success("**SOLUSI VALID** ✅")
                    st.write("Tidak melebihi kapasitas")
                else:
                    st.error("**MELEBIHI KAPASITAS** ❌")
                    st.write("Dikenai penalty")
                
                if best_list[-1] == best_list[-10] if len(best_list) > 10 else True:
                    st.success("**KONVERGEN** ✅")
                    st.write("Algoritma telah stabil")
                else:
                    st.warning("**BELUM KONVERGEN** ⚠️")
                    st.write("Perlu lebih banyak generasi")

else:
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Perhatian:</strong> Silakan lengkapi input data barang terlebih dahulu.
    </div>
    """, unsafe_allow_html=True)

# ============================
# FOOTER
# ============================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;'>"
    "Dibuat dengan ❤️ menggunakan Streamlit | Implementasi Genetic Algorithm untuk Knapsack Problem"
    "</div>",
    unsafe_allow_html=True
)