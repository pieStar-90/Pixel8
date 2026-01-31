import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass
from typing import List, Literal

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class SyntheticPost:
    id: str
    text: str
    timestamp: datetime
    channel: str
    sentiment: Literal['positive', 'negative', 'neutral']
    topic: str

@dataclass
class Signal:
    id: str
    title: str
    category: Literal['service', 'sentiment', 'fraud', 'misinformation']
    priority: Literal['high', 'medium', 'low']
    severity: float
    confidence: int
    status: Literal['new', 'in_review', 'resolved']
    created_at: datetime
    updated_at: datetime
    volume_change: int
    mention_count: int
    sentiment_change: int = None
    description: str = ""
    impact: dict = None
    drivers: List[dict] = None
    evidence: List[SyntheticPost] = None
    uncertainties: List[str] = None
    assigned_to: str = None
    reviewer: str = None
    notes: List[dict] = None

# ============================================================================
# SYNTHETIC DATA
# ============================================================================

def get_synthetic_posts():
    """Generate synthetic social media posts"""
    posts = [
        # App Login Issues
        SyntheticPost(
            id='post-1',
            text='App keeps freezing on the login screen. Been trying for 30 minutes now!',
            timestamp=datetime(2026, 1, 30, 14, 15),
            channel='Social Media',
            sentiment='negative',
            topic='login_issues'
        ),
        SyntheticPost(
            id='post-2',
            text='Anyone else having trouble logging into the app? It just crashes immediately.',
            timestamp=datetime(2026, 1, 30, 14, 28),
            channel='Social Media',
            sentiment='negative',
            topic='login_issues'
        ),
        SyntheticPost(
            id='post-3',
            text='Third time trying to log in today. Still not working. What\'s going on?',
            timestamp=datetime(2026, 1, 30, 14, 34),
            channel='Social Media',
            sentiment='negative',
            topic='login_issues'
        ),
        SyntheticPost(
            id='post-4',
            text='Cannot access my account since this morning. Need to pay bills urgently!',
            timestamp=datetime(2026, 1, 30, 14, 42),
            channel='Social Media',
            sentiment='negative',
            topic='login_issues'
        ),
        SyntheticPost(
            id='post-5',
            text='The mobile app login is broken. Stuck on the loading screen.',
            timestamp=datetime(2026, 1, 30, 14, 50),
            channel='Social Media',
            sentiment='negative',
            topic='login_issues'
        ),
        
        # Customer Service Sentiment
        SyntheticPost(
            id='post-6',
            text='Waited 45 minutes on hold with customer service. Still no resolution.',
            timestamp=datetime(2026, 1, 30, 10, 20),
            channel='Social Media',
            sentiment='negative',
            topic='customer_service'
        ),
        SyntheticPost(
            id='post-7',
            text='Customer service agent was not helpful at all. Very disappointed.',
            timestamp=datetime(2026, 1, 30, 11, 15),
            channel='Social Media',
            sentiment='negative',
            topic='customer_service'
        ),
        SyntheticPost(
            id='post-8',
            text='Been trying to reach customer support for days. No response.',
            timestamp=datetime(2026, 1, 29, 16, 30),
            channel='Social Media',
            sentiment='negative',
            topic='customer_service'
        ),
        SyntheticPost(
            id='post-9',
            text='The service quality has really declined lately. What happened?',
            timestamp=datetime(2026, 1, 29, 14, 20),
            channel='Social Media',
            sentiment='negative',
            topic='customer_service'
        ),
        
        # Scam Awareness (Normal)
        SyntheticPost(
            id='post-10',
            text='PSA: Always verify before clicking links claiming to be from your bank.',
            timestamp=datetime(2026, 1, 29, 12, 0),
            channel='Social Media',
            sentiment='neutral',
            topic='scam_awareness'
        ),
        SyntheticPost(
            id='post-11',
            text='Reminder: Banks never ask for your PIN via email or SMS.',
            timestamp=datetime(2026, 1, 29, 9, 30),
            channel='Social Media',
            sentiment='neutral',
            topic='scam_awareness'
        ),
        SyntheticPost(
            id='post-12',
            text='Stay safe online! Don\'t share your banking credentials with anyone.',
            timestamp=datetime(2026, 1, 28, 15, 45),
            channel='Social Media',
            sentiment='neutral',
            topic='scam_awareness'
        )
    ]
    return posts

def get_signals():
    """Generate signals from synthetic data"""
    posts = get_synthetic_posts()
    
    signals = [
        Signal(
            id='signal-1',
            title='App Login Issues - Sudden Spike',
            category='service',
            priority='high',
            severity=8.5,
            confidence=82,
            status='new',
            created_at=datetime(2026, 1, 30, 14, 34),
            updated_at=datetime(2026, 1, 30, 16, 12),
            volume_change=245,
            mention_count=47,
            description='Detected sudden 245% increase in complaints about mobile banking login failures starting at 11:30 AM. 47 customer mentions within 3 hours.',
            impact={
                'customer': 'high',
                'brand': 'medium',
                'operational': 'high'
            },
            drivers=[
                {'text': 'Login screen freezing', 'count': 31},
                {'text': 'Cannot access account', 'count': 18},
                {'text': 'App crashing on startup', 'count': 12}
            ],
            evidence=[p for p in posts if p.topic == 'login_issues'][:5],
            uncertainties=[
                'Sample size: Moderate (47 posts) - more data would increase confidence',
                'Time window: Short (3 hours) - trend may stabilize or escalate',
                'Root cause: Unknown - internal systems check recommended'
            ]
        ),
        Signal(
            id='signal-2',
            title='Customer Service Sentiment Decline',
            category='sentiment',
            priority='medium',
            severity=6.2,
            confidence=71,
            status='in_review',
            created_at=datetime(2026, 1, 30, 9, 15),
            updated_at=datetime(2026, 1, 30, 10, 30),
            volume_change=35,
            mention_count=89,
            sentiment_change=-15,
            description='Gradual 15% decrease in positive sentiment regarding customer service over the past week. 89 mentions with consistent negative themes.',
            impact={
                'customer': 'medium',
                'brand': 'medium',
                'operational': 'low'
            },
            drivers=[
                {'text': 'Long wait times', 'count': 34},
                {'text': 'Unhelpful responses', 'count': 28},
                {'text': 'No follow-up', 'count': 27}
            ],
            evidence=[p for p in posts if p.topic == 'customer_service'][:4],
            uncertainties=[
                'Sentiment analysis confidence: 71% - some ambiguous posts',
                'Baseline comparison: Week-over-week trend, seasonal factors unknown',
                'Attribution: Multiple potential causes require investigation'
            ],
            assigned_to='Communications Team',
            reviewer='Sarah Johnson',
            notes=[
                {
                    'text': 'Contacted customer service team. Investigating recent policy changes that may have triggered negative feedback. Will update in 2 hours.',
                    'author': 'Sarah Johnson',
                    'timestamp': datetime(2026, 1, 30, 10, 30)
                }
            ]
        ),
        Signal(
            id='signal-3',
            title='Scam Warning Posts - Routine Volume',
            category='fraud',
            priority='low',
            severity=3.1,
            confidence=65,
            status='resolved',
            created_at=datetime(2026, 1, 29, 15, 20),
            updated_at=datetime(2026, 1, 29, 18, 45),
            volume_change=5,
            mention_count=12,
            description='Routine customer awareness posts about scam prevention. Volume within expected baseline.',
            impact={
                'customer': 'low',
                'brand': 'low',
                'operational': 'low'
            },
            drivers=[
                {'text': 'General scam awareness', 'count': 8},
                {'text': 'Phishing warnings', 'count': 4}
            ],
            evidence=[p for p in posts if p.topic == 'scam_awareness'],
            uncertainties=[
                'Small sample size (12 posts) - statistical significance limited',
                'Topic classification: Some posts may be misclassified'
            ],
            assigned_to='Risk & Compliance Team',
            reviewer='Mike Chen',
            notes=[
                {
                    'text': 'Verified these are routine customer awareness posts about general scam prevention. No specific threats identified. Volume within normal range.',
                    'author': 'Mike Chen',
                    'timestamp': datetime(2026, 1, 29, 18, 45)
                }
            ]
        )
    ]
    return signals

def get_trend_data():
    """Generate 7-day trend data"""
    data = {
        'date': ['Jan 24', 'Jan 25', 'Jan 26', 'Jan 27', 'Jan 28', 'Jan 29', 'Jan 30'],
        'Service Issues': [8, 6, 10, 12, 15, 18, 35],
        'Sentiment Shifts': [12, 15, 18, 14, 20, 25, 22],
        'Fraud Rumors': [3, 2, 4, 3, 5, 3, 4]
    }
    return pd.DataFrame(data)

# ============================================================================
# STYLING & CONFIGURATION
# ============================================================================

def set_page_config():
    """Configure Streamlit page"""
    st.set_page_config(
        page_title="Responsible AI Signal Dashboard - Mashreq Bank",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

def load_css():
    """Load custom CSS for Mashreq branding"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f9fafb 0%, #eff6ff 100%);
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Custom colors */
    .risk-high { color: #EF4444; }
    .risk-medium { color: #F59E0B; }
    .risk-low { color: #10B981; }
    .mashreq-primary { color: #0A4D68; }
    .mashreq-accent { color: #088395; }
    
    /* Card styling */
    .signal-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 4px solid;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    .signal-card-high { border-left-color: #EF4444; }
    .signal-card-medium { border-left-color: #F59E0B; }
    .signal-card-low { border-left-color: #10B981; }
    
    /* Buttons */
    .stButton > button {
        background: #088395;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background: #0A4D68;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Headers */
    h1 {
        color: #0A4D68;
        font-weight: 800;
    }
    
    h2 {
        color: #0A4D68;
        font-weight: 700;
    }
    
    h3 {
        color: #1f2937;
        font-weight: 600;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    
    .badge-high {
        background: #FEE2E2;
        color: #991B1B;
    }
    
    .badge-medium {
        background: #FEF3C7;
        color: #92400E;
    }
    
    .badge-low {
        background: #D1FAE5;
        color: #065F46;
    }
    
    .badge-new {
        background: #FEF3C7;
        color: #92400E;
    }
    
    .badge-in-review {
        background: #DBEAFE;
        color: #1E40AF;
    }
    
    .badge-resolved {
        background: #D1FAE5;
        color: #065F46;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_priority_emoji(priority: str) -> str:
    """Get emoji for priority level"""
    return {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(priority, '⚪')

def get_priority_color(priority: str) -> str:
    """Get color for priority level"""
    return {'high': '#EF4444', 'medium': '#F59E0B', 'low': '#10B981'}.get(priority, '#gray')

def get_impact_color(level: str) -> str:
    """Get color for impact level"""
    colors = {'high': 'risk-high', 'medium': 'risk-medium', 'low': 'risk-low'}
    return colors.get(level, '')

def format_time_ago(dt: datetime) -> str:
    """Format datetime as relative time"""
    now = datetime(2026, 1, 30, 16, 0)  # Fixed "now" for demo
    diff = now - dt
    
    if diff.total_seconds() < 3600:
        minutes = int(diff.total_seconds() / 60)
        return f"{minutes}m ago"
    elif diff.total_seconds() < 86400:
        hours = int(diff.total_seconds() / 3600)
        return f"{hours}h ago"
    else:
        days = int(diff.total_seconds() / 86400)
        return f"{days}d ago"

# ============================================================================
# DASHBOARD COMPONENTS
# ============================================================================

def render_header():
    """Render dashboard header"""
    col1, col2 = st.columns([6, 1])
    
    with col1:
        st.markdown("""
        <div style='display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;'>
            <div style='background: linear-gradient(135deg, #0A4D68 0%, #088395 100%); 
                        width: 50px; height: 50px; border-radius: 12px; 
                        display: flex; align-items: center; justify-content: center;'>
                <span style='font-size: 2rem;'>🛡️</span>
            </div>
            <div>
                <h1 style='margin: 0;'>Responsible AI Signal Dashboard</h1>
                <p style='margin: 0; color: #6b7280; font-size: 0.9rem;'>Real-time social signal intelligence</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: right; padding-top: 10px;'>
            <span style='background: #088395; color: white; padding: 0.5rem 1rem; 
                         border-radius: 8px; font-weight: 600; font-size: 0.9rem;'>
                👤 Admin
            </span>
        </div>
        """, unsafe_allow_html=True)

def render_metrics():
    """Render top-level metrics"""
    signals = get_signals()
    active_signals = [s for s in signals if s.status != 'resolved']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: white; padding: 1.5rem; border-radius: 12px; 
                    text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
            <div style='font-size: 3rem; margin-bottom: 0.5rem;'>🟡</div>
            <div style='font-size: 2rem; font-weight: 700; color: #F59E0B; margin-bottom: 0.25rem;'>MODERATE</div>
            <div style='color: #6b7280; font-size: 0.9rem; font-weight: 500;'>Overall Risk Level</div>
            <div style='color: #6b7280; font-size: 0.8rem; margin-top: 0.5rem;'>Confidence: 76%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background: white; padding: 1.5rem; border-radius: 12px; 
                    text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
            <div style='font-size: 3rem; margin-bottom: 0.5rem;'>📊</div>
            <div style='font-size: 2rem; font-weight: 700; color: #0A4D68; margin-bottom: 0.25rem;'>{len(signals)}</div>
            <div style='color: #6b7280; font-size: 0.9rem; font-weight: 500;'>Active Signals</div>
            <div style='color: #EF4444; font-size: 0.8rem; margin-top: 0.5rem;'>↑ +3 today</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        pending = len([s for s in signals if s.status == 'new'])
        st.markdown(f"""
        <div style='background: white; padding: 1.5rem; border-radius: 12px; 
                    text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
            <div style='font-size: 3rem; margin-bottom: 0.5rem;'>⏱️</div>
            <div style='font-size: 2rem; font-weight: 700; color: #0A4D68; margin-bottom: 0.25rem;'>{pending}</div>
            <div style='color: #6b7280; font-size: 0.9rem; font-weight: 500;'>Pending Reviews</div>
            <div style='color: #6b7280; font-size: 0.8rem; margin-top: 0.5rem;'>2 urgent</div>
        </div>
        """, unsafe_allow_html=True)

def render_trend_chart():
    """Render 7-day trend chart"""
    st.markdown("### 📊 Signal Trends (Last 7 Days)")
    
    df = get_trend_data()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'], 
        y=df['Service Issues'],
        mode='lines+markers',
        name='Service Issues',
        line=dict(color='#088395', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'], 
        y=df['Sentiment Shifts'],
        mode='lines+markers',
        name='Sentiment Shifts',
        line=dict(color='#FFA726', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'], 
        y=df['Fraud Rumors'],
        mode='lines+markers',
        name='Fraud Rumors',
        line=dict(color='#EF4444', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Poppins', size=12),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        height=350
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e5e7eb')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e5e7eb')
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("📈 **Trend Analysis:** Service-related signals showing significant increase on Jan 30. Sentiment remains elevated compared to baseline.")

def render_signal_distribution():
    """Render signal distribution bar charts - Mashreq style"""
    st.markdown("<h2>📊 Signal Distribution Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Priority Distribution
        priority_data = {
            'Priority': ['High', 'Medium', 'Low'],
            'Count': [3, 5, 4]
        }
        
        fig1 = go.Figure(data=[
            go.Bar(
                x=priority_data['Priority'],
                y=priority_data['Count'],
                marker=dict(
                    color=['#DC2626', '#F59E0B', '#10B981'],
                    line=dict(color='#ffffff', width=2)
                ),
                text=priority_data['Count'],
                textposition='auto',
                textfont=dict(size=14, color='white', family='Roboto', weight='bold')
            )
        ])
        
        fig1.update_layout(
            title=dict(
                text='Signals by Priority Level',
                font=dict(size=16, color='#00539F', family='Roboto', weight='bold')
            ),
            plot_bgcolor='#F8FAFC',
            paper_bgcolor='white',
            font=dict(family='Roboto', size=12, color='#1E293B'),
            margin=dict(l=20, r=20, t=60, b=40),
            height=350,
            xaxis=dict(
                showgrid=False,
                title=dict(text='Priority Level', font=dict(size=12, color='#64748B'))
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='#E2E8F0',
                title=dict(text='Number of Signals', font=dict(size=12, color='#64748B'))
            )
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Category Distribution
        category_data = {
            'Category': ['Service', 'Sentiment', 'Fraud', 'Misinformation'],
            'Count': [5, 4, 2, 1]
        }
        
        fig2 = go.Figure(data=[
            go.Bar(
                x=category_data['Category'],
                y=category_data['Count'],
                marker=dict(
                    color=['#00539F', '#0066CC', '#00A3E0', '#003366'],
                    line=dict(color='#ffffff', width=2)
                ),
                text=category_data['Count'],
                textposition='auto',
                textfont=dict(size=14, color='white', family='Roboto', weight='bold')
            )
        ])
        
        fig2.update_layout(
            title=dict(
                text='Signals by Category',
                font=dict(size=16, color='#00539F', family='Roboto', weight='bold')
            ),
            plot_bgcolor='#F8FAFC',
            paper_bgcolor='white',
            font=dict(family='Roboto', size=12, color='#1E293B'),
            margin=dict(l=20, r=20, t=60, b=40),
            height=350,
            xaxis=dict(
                showgrid=False,
                title=dict(text='Category', font=dict(size=12, color='#64748B'))
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='#E2E8F0',
                title=dict(text='Number of Signals', font=dict(size=12, color='#64748B'))
            )
        )
        
        st.plotly_chart(fig2, use_container_width=True)

def render_horizontal_bar_chart():
    """Render horizontal bar chart for top drivers - Mashreq style"""
    st.markdown("<h2>🔍 Top Signal Drivers</h2>", unsafe_allow_html=True)
    
    drivers_data = {
        'Driver': ['Login screen freezing', 'Long wait times', 'Cannot access account', 
                   'Unhelpful responses', 'App crashing on startup', 'No follow-up'],
        'Mentions': [31, 34, 18, 28, 12, 27]
    }
    
    fig = go.Figure(data=[
        go.Bar(
            y=drivers_data['Driver'],
            x=drivers_data['Mentions'],
            orientation='h',
            marker=dict(
                color=drivers_data['Mentions'],
                colorscale=[
                    [0, '#D1FAE5'],
                    [0.5, '#00A3E0'],
                    [1, '#00539F']
                ],
                line=dict(color='#ffffff', width=2)
            ),
            text=drivers_data['Mentions'],
            textposition='auto',
            textfont=dict(size=13, color='white', family='Roboto', weight='bold')
        )
    ])
    
    fig.update_layout(
        plot_bgcolor='#F8FAFC',
        paper_bgcolor='white',
        font=dict(family='Roboto', size=12, color='#1E293B'),
        margin=dict(l=20, r=20, t=20, b=40),
        height=400,
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#E2E8F0',
            title=dict(text='Number of Mentions', font=dict(size=12, color='#64748B'))
        ),
        yaxis=dict(
            showgrid=False,
            title=None
        ),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
def render_signal_card(signal: Signal):
    """Render individual signal card - SIMPLE TEST VERSION"""
    priority_emoji = get_priority_emoji(signal.priority)
    time_ago = format_time_ago(signal.created_at)
    
    st.write(f"{priority_emoji} **{signal.title}**")
    st.write(f"Priority: {signal.priority.upper()} | Time: {time_ago}")
    st.write(f"Severity: {signal.severity}/10 | Confidence: {signal.confidence}%")
    st.write(f"Volume: +{signal.volume_change}% | Mentions: {signal.mention_count}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"View Details →", key=f"view_{signal.id}"):
            st.session_state.selected_signal = signal.id
            st.session_state.page = "detail"
            st.rerun()
    with col2:
        if st.button(f"Assign to Team", key=f"assign_{signal.id}"):
            st.session_state.page = "review"
            st.rerun()
    
    st.markdown("---")

def render_dashboard():
    """Render main dashboard page"""
    render_header()
    st.markdown("---")
    
    # Metrics
    render_metrics()
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Trend Chart
    render_trend_chart()
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Bar Charts 
    render_signal_distribution()
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    render_horizontal_bar_chart()
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Signal Cards
    st.markdown("### 🔍 Top Priority Signals")
    
    signals = get_signals()
    active_signals = [s for s in signals if s.status != 'resolved']
    
    for signal in active_signals:
        render_signal_card(signal)
    
    # Recent Activity
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    ### 📋 Recent Activity
    
    - 🔵 Operations Team assigned "App Login Issues" - *15m ago*
    - 🔵 Communications reviewed "Brand Sentiment" - *1h ago*
    - 🟢 Signal "Payment Delays" marked Resolved - - *3h ago*
    """)

def render_signal_detail():
    """Render signal detail page"""
    signal_id = st.session_state.get('selected_signal', 'signal-1')
    signals = get_signals()
    signal = next((s for s in signals if s.id == signal_id), signals[0])
    
    # Back button
    if st.button("← Back to Dashboard"):
        st.session_state.page = "dashboard"
        st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Header
    priority_emoji = get_priority_emoji(signal.priority)
    st.markdown(f"""
    <div style='background: white; padding: 2rem; border-radius: 12px; 
                border-left: 6px solid {get_priority_color(signal.priority)}; 
                box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 2rem;'>
        <div style='display: flex; justify-content: space-between; align-items: start; margin-bottom: 1.5rem;'>
            <div style='display: flex; align-items: center; gap: 1rem;'>
                <span style='font-size: 2.5rem;'>{priority_emoji}</span>
                <span class='badge badge-{signal.priority}' style='font-size: 0.9rem; padding: 0.5rem 1rem;'>
                    {signal.priority.upper()} PRIORITY SIGNAL
                </span>
            </div>
            <div style='text-align: right; color: #6b7280; font-size: 0.875rem;'>
                <div>Detected: {signal.created_at.strftime('%b %d, %Y at %I:%M %p')}</div>
                <div>Last Updated: {signal.updated_at.strftime('%b %d, %Y at %I:%M %p')}</div>
            </div>
        </div>
        
        <h1 style='color: #1f2937; margin-bottom: 2rem;'>{signal.title}</h1>
        
        <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;'>
            <div style='background: #FEF2F2; border: 2px solid #FCA5A5; border-radius: 8px; padding: 1.5rem; text-align: center;'>
                <div style='font-size: 2.5rem; font-weight: 700; color: #EF4444;'>{signal.severity}</div>
                <div style='color: #6b7280; font-size: 0.875rem; margin-top: 0.5rem;'>Severity / 10</div>
                <div style='color: #EF4444; font-size: 0.75rem; margin-top: 0.5rem;'>🔴 Critical Level</div>
            </div>
            <div style='background: #F0FDF4; border: 2px solid #86EFAC; border-radius: 8px; padding: 1.5rem; text-align: center;'>
                <div style='font-size: 2.5rem; font-weight: 700; color: #10B981;'>{signal.confidence}%</div>
                <div style='color: #6b7280; font-size: 0.875rem; margin-top: 0.5rem;'>Confidence</div>
                <div style='color: #10B981; font-size: 0.75rem; margin-top: 0.5rem;'>🟢 High Certainty</div>
            </div>
            <div style='background: #FFFBEB; border: 2px solid #FCD34D; border-radius: 8px; padding: 1.5rem; text-align: center;'>
                <div style='font-size: 2.5rem; font-weight: 700;'>⚡</div>
                <div style='color: #6b7280; font-size: 0.875rem; margin-top: 0.5rem;'>Urgency</div>
                <div style='color: #F59E0B; font-size: 0.75rem; margin-top: 0.5rem;'>HIGH</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Why This Matters - USE ST.WRITE INSTEAD
    st.markdown("### 💡 Why This Matters")
    st.markdown(signal.description, unsafe_allow_html=True)

    
    st.markdown("**POTENTIAL IMPACT:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        impact_class = get_impact_color(signal.impact["customer"])
        st.markdown(f"**Customer Experience:** <span class='{impact_class}'>{signal.impact['customer'].upper()}</span>", unsafe_allow_html=True)
    with col2:
        impact_class = get_impact_color(signal.impact["brand"])
        st.markdown(f"**Brand Reputation:** <span class='{impact_class}'>{signal.impact['brand'].upper()}</span>", unsafe_allow_html=True)
    with col3:
        impact_class = get_impact_color(signal.impact["operational"])
        st.markdown(f"**Operational Risk:** <span class='{impact_class}'>{signal.impact['operational'].upper()}</span>", unsafe_allow_html=True)
    
    st.info("**RECOMMENDED ACTION:** Route to Operations & IT Teams for immediate review and investigation.")
    
    # Signal Analysis
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📈 Signal Analysis")
    
    st.write("**DETECTION METHOD:** Volume spike detection (last 3 hours vs. baseline)")
    st.write("**KEY METRICS:**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"📈 Volume Increase: **<span style='color: #EF4444;'>+{signal.volume_change}%</span>**", unsafe_allow_html=True)
    with col2:
        st.markdown(f"💬 Total Mentions: **{signal.mention_count}**")
    
    st.markdown("**DRIVERS IDENTIFIED:**")
    for i, driver in enumerate(signal.drivers, 1):
        st.markdown(f"{i}. {driver['text']} - **{driver['count']} mentions**")
    
    # Evidence
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📝 Evidence & Examples")
    
    for i, post in enumerate(signal.evidence[:3], 1):
        sentiment_color = {'negative': '#EF4444', 'positive': '#10B981', 'neutral': '#6B7280'}
        st.markdown(f"""
        <div style='background: #F9FAFB; border-left: 4px solid #088395; 
                    border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem;'>
            <div style='display: flex; justify-content: between; margin-bottom: 0.5rem;'>
                <span class='badge' style='background: #E5E7EB; color: #374151;'>Synthetic Post #{i}</span>
                <span style='color: #9CA3AF; font-size: 0.75rem; margin-left: auto;'>
                    {post.timestamp.strftime('%b %d, %Y at %I:%M %p')}
                </span>
            </div>
            <p style='font-style: italic; color: #374151; margin: 1rem 0;'>"{post.text}"</p>
            <div style='display: flex; gap: 0.5rem; flex-wrap: wrap;'>
                <span class='badge' style='background: #DBEAFE; color: #1E40AF;'>{post.channel}</span>
                <span class='badge' style='background: {"#FEE2E2" if post.sentiment == "negative" else "#D1FAE5"}; 
                                         color: {sentiment_color[post.sentiment]};'>
                    Sentiment: {post.sentiment}
                </span>
                <span class='badge' style='background: #F3E8FF; color: #6B21A8;'>
                    Topic: {post.topic.replace('_', ' ')}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Uncertainty Factors
    st.markdown("<br>", unsafe_allow_html=True)
    st.warning("⚠️ **Uncertainty Factors**")
    for uncertainty in signal.uncertainties:
        st.markdown(f"- {uncertainty}")
    
    st.markdown(f"""
    **CONFIDENCE BREAKDOWN:**
    - Pattern Recognition: **91%** ✓
    - Volume Significance: **85%** ✓
    - Sentiment Consistency: **78%** ~
    - **Overall: {signal.confidence}%**
    """)
    
    # Actions
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎯 Human Actions")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("Assign to Operations Team", use_container_width=True)
    with col2:
        st.button("Assign to IT Support", use_container_width=True)
    with col3:
        st.button("Mark for Communications", use_container_width=True)

def render_review_queue():
    """Render review queue page"""
    st.markdown("# Human Review Queue")
    st.markdown("Assign, review, and resolve signals")
    
    # Status tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        f"All ({len(get_signals())})",
        f"New ({len([s for s in get_signals() if s.status == 'new'])})",
        f"In Review ({len([s for s in get_signals() if s.status == 'in_review'])})",
        f"Resolved ({len([s for s in get_signals() if s.status == 'resolved'])})"
    ])
    
    signals = get_signals()
    
    with tab1:
        for signal in signals:
            render_review_card(signal)
    
    with tab2:
        for signal in [s for s in signals if s.status == 'new']:
            render_review_card(signal)
    
    with tab3:
        for signal in [s for s in signals if s.status == 'in_review']:
            render_review_card(signal)
    
    with tab4:
        for signal in [s for s in signals if s.status == 'resolved']:
            render_review_card(signal)
    
    # Queue Stats
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Queue Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Review Time", "2.5 hours")
    with col2:
        st.metric("Resolution Rate", "85% within 24h")
    with col3:
        st.metric("False Positive Rate", "12%")

def render_review_card(signal: Signal):
    """Render review queue card"""
    priority_emoji = get_priority_emoji(signal.priority)
    
    st.markdown(f"""
    <div class='signal-card signal-card-{signal.priority}'>
        <div style='display: flex; justify-content: space-between; margin-bottom: 1rem;'>
            <div style='display: flex; align-items: center; gap: 0.75rem;'>
                <span style='font-size: 1.5rem;'>{priority_emoji}</span>
                <span class='badge badge-{signal.status}'>{signal.status.replace('_', ' ').upper()}</span>
            </div>
            <span style='color: #9ca3af; font-size: 0.875rem;'>
                Created: {signal.created_at.strftime('%b %d, %Y at %I:%M %p')}
            </span>
        </div>
        
        <h3 style='margin-bottom: 1rem;'>{signal.title}</h3>
        
        <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-bottom: 1rem;'>
            <div><span style='color: #6b7280;'>Severity:</span> <strong>{signal.severity}/10</strong></div>
            <div><span style='color: #6b7280;'>Confidence:</span> <strong>{signal.confidence}%</strong></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create unique container for each signal
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox(
                "Assigned to:",
                ["Select Team", "Operations Team", "Communications Team", "IT Support", "Risk & Compliance"],
                key=f"assign_select_{signal.id}_{signal.created_at.timestamp()}"  # Added timestamp for uniqueness
            )
        with col2:
            st.selectbox(
                "Status:",
                ["NEW", "IN REVIEW", "RESOLVED"],
                index=["new", "in_review", "resolved"].index(signal.status),
                key=f"status_select_{signal.id}_{signal.created_at.timestamp()}"  # Added timestamp for uniqueness
            )
        
        if signal.reviewer:
            st.markdown(f"**Reviewer:** {signal.reviewer}")
        
        notes_text = signal.notes[0]['text'] if signal.notes else ""
        st.text_area(
            "📝 Review Notes (Internal Use)",
            value=notes_text,
            height=100,
            key=f"notes_{signal.id}_{signal.created_at.timestamp()}",  # Added timestamp for uniqueness
            placeholder="Add your assessment, findings, or actions taken..."
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button("View Full Details", key=f"review_view_{signal.id}_{signal.created_at.timestamp()}")
        with col2:
            st.button("Update Status", key=f"review_update_{signal.id}_{signal.created_at.timestamp()}")
        with col3:
            st.button("Add Note", key=f"review_note_{signal.id}_{signal.created_at.timestamp()}")
    
    st.markdown("---")
    
def render_executive_briefing():
    """Render executive briefing page"""
    st.markdown("# 📑 Executive Insight Briefing")
    st.markdown(f"**Generated:** Friday, January 30, 2026 at 4:00 PM")
    st.markdown(f"**Report Period:** Last 7 Days (Jan 23 - Jan 30, 2026)")
    
    st.markdown("---")
    
    # Risk Overview
    st.markdown("## 📊 Risk Overview")
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%); 
                border: 2px solid #F59E0B; border-radius: 12px; padding: 2rem; margin-bottom: 2rem;'>
        <div style='display: flex; align-items: center; gap: 1.5rem; margin-bottom: 1.5rem;'>
            <span style='font-size: 4rem;'>🟡</span>
            <div>
                <div style='font-size: 2rem; font-weight: 700; color: #1f2937;'>MODERATE RISK STATUS</div>
                <div style='color: #6b7280;'>Confidence Level: 76%</div>
            </div>
        </div>
        
        <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1rem;'>
            <div style='background: white; border: 2px solid #FCA5A5; border-radius: 8px; padding: 1rem; text-align: center;'>
                <div style='font-size: 2rem; font-weight: 700; color: #EF4444;'>3</div>
                <div style='color: #6b7280; font-size: 0.875rem;'>High Priority</div>
            </div>
            <div style='background: white; border: 2px solid #FCD34D; border-radius: 8px; padding: 1rem; text-align: center;'>
                <div style='font-size: 2rem; font-weight: 700; color: #F59E0B;'>5</div>
                <div style='color: #6b7280; font-size: 0.875rem;'>Medium Priority</div>
            </div>
            <div style='background: white; border: 2px solid #86EFAC; border-radius: 8px; padding: 1rem; text-align: center;'>
                <div style='font-size: 2rem; font-weight: 700; color: #10B981;'>4</div>
                <div style='color: #6b7280; font-size: 0.875rem;'>Low Priority</div>
            </div>
        </div>
        
        <div style='color: #374151; font-size: 0.9rem;'>
            📈 Trend: ↗ Increased activity vs. previous week (+18%)
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Insights
    st.markdown("## 🎯 Key Insights (Top 3)")
    
    # Insight 1
    st.markdown("""
    ### 1️⃣ SERVICE DISRUPTION DETECTED
    
    **App Login Issues - Sudden Spike** 🔴 HIGH PRIORITY
    
    **SITUATION:**  
    Detected sudden 245% increase in complaints about mobile banking login failures starting at 11:30 AM. 
    47 customer mentions within 3 hours.
    
    **POTENTIAL IMPACT:**
    - Immediate: Customer access disruption
    - Short-term: Increased call center volume
    - Medium-term: Brand reputation if unresolved
    
    **CONFIDENCE:** 82% (High volume, clear pattern)
    
    **RECOMMENDED ACTION:**
    - ✅ Immediate technical investigation required
    - ✅ Communications team to prepare customer update
    - ✅ Status: Assigned to Operations (In Progress)
    """)
    
    st.markdown("---")
    
    # Insight 2
    st.markdown("""
    ### 2️⃣ BRAND SENTIMENT SHIFT
    
    **Customer Service Sentiment Decline** 🟡 MEDIUM PRIORITY
    
    **SITUATION:**  
    Gradual 15% decrease in positive sentiment regarding customer service over the past week. 
    89 mentions with consistent negative themes.
    
    **POTENTIAL IMPACT:**
    - Customer satisfaction metrics at risk
    - Possible operational quality issue
    - Early warning for service improvements needed
    
    **CONFIDENCE:** 71% (Moderate sample, clear trend)
    
    **RECOMMENDED ACTION:**
    - ✅ Review customer service protocols
    - ✅ Analyze specific complaint drivers
    - ✅ Status: Under Review by Communications Team
    """)
    
    st.markdown("---")
    
    # Insight 3
    st.markdown("""
    ### 3️⃣ BASELINE MONITORING
    
    **Fraud/Scam Discussion - Normal Levels** 🟢 LOW PRIORITY
    
    **SITUATION:**  
    Routine customer awareness posts about scam prevention. Volume within expected baseline (12 mentions).
    
    **ASSESSMENT:**  
    No immediate action required. Standard monitoring continues. This represents healthy customer vigilance.
    
    **CONFIDENCE:** 65% (Small sample, normal pattern)
    """)
    
    # Trend Analysis
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## 📈 Trend Analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Signals", "12", "+20%", help="Previous: 10")
    with col2:
        st.metric("High Priority", "3", "+200%", help="Previous: 1")
    with col3:
        st.metric("Avg Confidence", "76%", "+3pp", help="Previous: 73%")
    
    st.markdown("""
    **EMERGING PATTERNS:**
    - Service-related issues increasing
    - Sentiment metrics showing minor decline
    - Fraud discussions remain stable
    """)
    
    # Limitations
    st.markdown("<br>", unsafe_allow_html=True)
    st.warning("""
    ⚠️ **Limitations & Uncertainties**
    
    - All signals based on synthetic public data
    - Small sample sizes may limit statistical confidence
    - Root causes require human investigation
    - Automated recommendations are advisory only
    - System does not monitor individuals or take actions
    
    ℹ️ **This report is generated to support human decision-making.** All actions require executive approval.
    """)
    
    # Recommended Actions
    st.markdown("## 🎯 Recommended Next Steps")
    
    st.error("""
    **1. IMMEDIATE (Next 4 Hours):**
    - Technical review of app login system
    - Customer communication strategy
    """)
    
    st.warning("""
    **2. SHORT-TERM (Next 24 Hours):**
    - Customer service quality audit
    - Sentiment driver analysis
    """)
    
    st.info("""
    **3. ONGOING:**
    - Continue monitoring all active signals
    - Daily executive briefing updates
    """)
    
    # Export Options
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("📧 Email This Report", use_container_width=True)
    with col2:
        st.button("📄 Export PDF", use_container_width=True)
    with col3:
        st.button("⚙️ Settings", use_container_width=True)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point"""
    # Configuration
    set_page_config()
    load_css()
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'dashboard'
    
    # Navigation
    st.markdown("""
    <div style='background: white; border-bottom: 2px solid #e5e7eb; 
                padding: 1rem 2rem; margin: -1rem -1rem 2rem -1rem; 
                display: flex; gap: 0.5rem;'>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns([2, 2, 2, 6])
    
    with col1:
        if st.button("📊 Dashboard", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()
    
    with col2:
        if st.button("📋 Review Queue", use_container_width=True):
            st.session_state.page = "review"
            st.rerun()
    
    with col3:
        if st.button("📑 Executive Briefing", use_container_width=True):
            st.session_state.page = "briefing"
            st.rerun()
    
    # Render current page
    if st.session_state.page == 'dashboard':
        render_dashboard()
    elif st.session_state.page == 'detail':
        render_signal_detail()
    elif st.session_state.page == 'review':
        render_review_queue()
    elif st.session_state.page == 'briefing':
        render_executive_briefing()

if __name__ == "__main__":
    main()
