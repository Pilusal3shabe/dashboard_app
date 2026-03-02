import streamlit as st
import base64
from pathlib import Path

# Tlowana Colors
TLOWANA_GREEN = '#6B7C2E'
TLOWANA_DARK = '#2C2C2C'

# Page config
st.set_page_config(page_title="Tlowana Test", layout="wide")

# Custom CSS
st.markdown(f"""
    <style>
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {TLOWANA_GREEN} 0%, {TLOWANA_DARK} 100%);
    }}
    [data-testid="stSidebar"] * {{
        color: white !important;
    }}
    .logo-container {{
        text-align: center;
        padding: 10px;
        background: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }}
    .main-header {{
        background: linear-gradient(135deg, {TLOWANA_GREEN} 0%, {TLOWANA_DARK} 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }}
    </style>
""", unsafe_allow_html=True)

# Logo function
def display_logo():
    try:
        logo_path = Path("Company_Logo.jpg")
        if logo_path.exists():
            with open(logo_path, "rb") as f:
                logo_data = base64.b64encode(f.read()).decode()
                st.markdown(
                    f'<div class="logo-container"><img src="data:image/jpeg;base64,{logo_data}" width="250"></div>',
                    unsafe_allow_html=True
                )
                return True
        else:
            st.markdown(
                f'<div class="logo-container"><h2 style="color: {TLOWANA_GREEN};">TLOWANA RESOURCES</h2></div>',
                unsafe_allow_html=True
            )
            return False
    except Exception as e:
        st.markdown(
            f'<div class="logo-container"><h2 style="color: {TLOWANA_GREEN};">TLOWANA RESOURCES</h2><p style="color:red;">Error: {str(e)}</p></div>',
            unsafe_allow_html=True
        )
        return False

# Sidebar
with st.sidebar:
    logo_found = display_logo()
    st.markdown("---")
    st.title("Test Dashboard")
    
    if logo_found:
        st.success("✅ Logo loaded successfully!")
    else:
        st.error("❌ Logo not found. Check Company_Logo.jpg is in same folder.")
    
    st.info(f"Current directory: {Path.cwd()}")
    st.info(f"Logo path: {Path('Company_Logo.jpg').absolute()}")
    st.info(f"Logo exists: {Path('Company_Logo.jpg').exists()}")

# Main content
st.markdown(f"""
    <div class="main-header">
        <h1>🔥 Tlowana Resources Branding Test</h1>
        <p>Testing logo and color theme</p>
    </div>
""", unsafe_allow_html=True)

st.write("## Branding Checklist:")
col1, col2 = st.columns(2)

with col1:
    st.write("**Expected:**")
    st.write("✅ Logo in sidebar (if file found)")
    st.write("✅ Green gradient sidebar")
    st.write("✅ White text in sidebar")
    st.write("✅ Green gradient header")
    st.write("✅ Company colors throughout")

with col2:
    st.write("**Logo Status:**")
    if logo_found:
        st.success("Logo is displaying correctly!")
    else:
        st.error("Logo NOT found. Please ensure:")
        st.write("1. File named exactly: **Company_Logo.jpg**")
        st.write("2. File in same folder as this script")
        st.write("3. File has read permissions")

st.markdown("---")
st.write("### Color Theme Test")

st.markdown(f"""
<div style="display: flex; gap: 20px; margin: 20px 0;">
    <div style="background: {TLOWANA_GREEN}; padding: 20px; border-radius: 10px; color: white; flex: 1;">
        <strong>Primary Green</strong><br>{TLOWANA_GREEN}
    </div>
    <div style="background: {TLOWANA_DARK}; padding: 20px; border-radius: 10px; color: white; flex: 1;">
        <strong>Dark</strong><br>{TLOWANA_DARK}
    </div>
</div>
""", unsafe_allow_html=True)

st.success("If you can see the logo above and green colors, branding is working!")
