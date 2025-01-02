import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Set page config for a wider layout and custom title
st.set_page_config(
    page_title="S&P 500 Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .stPlotlyChart {
        background-color: #0E1117;
    }
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .sidebar .sidebar-content {
        background-color: #0E1117;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess S&P 500 data"""
    df = pd.read_csv('data\\sp500_companies.csv')
    df['MarketCapBillions'] = df['Marketcap'] / 1e9
    return df

def format_large_number(num):
    """Format large numbers into readable string with B/M suffix"""
    if num >= 1e9:
        return f"${num/1e9:.1f}B"
    elif num >= 1e6:
        return f"${num/1e6:.1f}M"
    else:
        return f"${num:,.0f}"

def create_sector_chart(df):
    """Create an interactive sector distribution chart"""
    sector_data = df.groupby('Sector')['MarketCapBillions'].sum().reset_index()
    fig = px.pie(
        sector_data,
        values='MarketCapBillions',
        names='Sector',
        title='Market Cap Distribution by Sector',
        color_discrete_sequence=px.colors.sequential.YlOrRd,
        hole=0.4
    )
    fig.update_layout(
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        font_color='#FAFAFA'
    )
    return fig

def create_performance_chart(df):
    """Create sector performance comparison chart"""
    sector_perf = df.groupby('Sector').agg({
        'Currentprice': 'mean',
        'MarketCapBillions': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sector_perf['Sector'],
        y=sector_perf['Currentprice'],
        name='Avg Price',
        marker_color='#FFD700'
    ))
    
    fig.update_layout(
        title='Sector Performance Overview',
        xaxis_tickangle=-45,
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        font_color='#FAFAFA',
        showlegend=True
    )
    return fig

def main():
    # Load data
    df = load_data()
    
    # Sidebar filters
    st.sidebar.header('Filters')
    selected_sector = st.sidebar.multiselect(
        'Select Sectors',
        options=df['Sector'].unique(),
        default=df['Sector'].unique()
    )
    
    min_market_cap = st.sidebar.number_input(
        'Minimum Market Cap (Billions)',
        min_value=0.0,
        max_value=float(df['MarketCapBillions'].max()),
        value=0.0
    )
    
    # Filter data
    mask = (
        df['Sector'].isin(selected_sector) &
        (df['MarketCapBillions'] >= min_market_cap)
    )
    filtered_df = df[mask]
    
    # Header with key metrics
    st.title('📈 S&P 500 Dashboard')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Total Market Cap",
            format_large_number(filtered_df['Marketcap'].sum())
        )
    with col2:
        st.metric(
            "Average Stock Price",
            f"${filtered_df['Currentprice'].mean():,.2f}"
        )
    with col3:
        st.metric(
            "Number of Companies",
            len(filtered_df)
        )
    
    # Charts section
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            create_sector_chart(filtered_df),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            create_performance_chart(filtered_df),
            use_container_width=True
        )
    
    # Companies table
    st.header('Top Companies by Market Cap')
    top_companies = filtered_df.nlargest(10, 'MarketCapBillions')[
        ['Symbol', 'Shortname', 'Sector', 'Currentprice', 'MarketCapBillions']
    ].reset_index(drop=True)
    
    # Format the MarketCapBillions column
    top_companies['Market Cap'] = top_companies['MarketCapBillions'].apply(
        lambda x: f"${x:,.1f}B"
    )
    top_companies['Current Price'] = top_companies['Currentprice'].apply(
        lambda x: f"${x:,.2f}"
    )
    
    st.dataframe(
        top_companies[['Symbol', 'Shortname', 'Sector', 'Current Price', 'Market Cap']],
        use_container_width=True,
        hide_index=True
    )

if __name__ == "__main__":
    main()