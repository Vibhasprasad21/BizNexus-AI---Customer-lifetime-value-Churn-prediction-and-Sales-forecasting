import streamlit as st
import streamlit.components.v1 as components
import base64
from PIL import Image
import pandas as pd
import time
import os

# Page configuration
st.set_page_config(
    page_title="BizNexus AI | Home",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
def load_css():
    css = """
    <style>
        /* Main styling with light pastel background */
        .main {
            background-color: #f4f6f9; /* Light pastel blue-gray background */
            color: #2c3e50; /* Dark text for readability */
        }
        
        /* Animation for tagline */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .animate-text {
            animation: fadeIn 1.5s ease-out;
        }
        
        /* Styling for sections */
        .header-container {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background-color: rgba(255, 255, 255, 0.7);
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }
        
        .hero-section {
            text-align: center;
            padding: 40px 20px;
            background: linear-gradient(135deg, #e8edf2 0%, #d1dbe7 100%);
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
        }
        
        /* Logo styling */
        .logo-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 0 auto;
            text-align: center;
            margin-bottom: 20px;
        }
        
        .logo-container img {
            width: 100px !important; /* Enlarged logo */
            margin: 0 auto;
            display: block;
        }
        
        .feature-box {
            background-color: rgba(255, 255, 255, 0.7);
            border-radius: 10px;
            padding: 20px;
            margin: 10px;
            text-align: center;
            transition: transform 0.3s, box-shadow 0.3s;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.06);
        }
        
        .feature-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        }
        
        .team-card {
            background-color: rgba(255, 255, 255, 0.7);
            border-radius: 10px;
            padding: 20px;
            margin: 10px;
            text-align: center;
            transition: transform 0.3s;
            height: 100%;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.06);
        }
        
        .team-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        }
        
        .social-icon {
            margin: 0 10px;
            color: #4da6ff;
            font-size: 20px;
        }
        
        .get-started-btn {
            background-color: #4da6ff;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 12px 30px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s;
            margin-top: 30px;
            font-weight: bold;
            display: inline-block;
            text-decoration: none;
        }
        
        .get-started-btn:hover {
            background-color: #3d86cc;
            transform: translateY(-2px);
            box-shadow: 0 8px 15px rgba(77, 166, 255, 0.3);
        }
        
        /* For the background particles animation */
        #tsparticles {
            position: fixed;
            width: 100%;
            height: 100%;
            top: 0;
            left: 0;
            z-index: -1;
        }
        
        /* Logo animation */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .logo-animate {
            animation: pulse 3s infinite;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .header-container {
                flex-direction: column;
            }
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Load animated background with particles.js
def load_particles_bg():
    particles_js = """
    <div id="tsparticles"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/tsparticles/2.9.3/tsparticles.bundle.min.js"></script>
    <script>
        tsParticles.load("tsparticles", {
            fpsLimit: 60,
            particles: {
                number: {
                    value: 80,
                    density: {
                        enable: true,
                        value_area: 800
                    }
                },
                color: {
                    value: "#4da6ff"
                },
                shape: {
                    type: "circle"
                },
                opacity: {
                    value: 0.3,
                    random: true
                },
                size: {
                    value: 3,
                    random: true
                },
                move: {
                    enable: true,
                    speed: 1,
                    direction: "none",
                    random: true,
                    straight: false,
                    out_mode: "out",
                    bounce: false,
                }
            },
            interactivity: {
                detect_on: "canvas",
                events: {
                    onhover: {
                        enable: true,
                        mode: "grab"
                    },
                    onclick: {
                        enable: true,
                        mode: "push"
                    },
                    resize: true
                },
                modes: {
                    grab: {
                        distance: 140,
                        line_linked: {
                            opacity: 0.5
                        }
                    },
                    push: {
                        particles_nb: 4
                    }
                }
            },
            retina_detect: true
        });
    </script>
    """
    components.html(particles_js, height=0)

# Get logo - Fixed path handling and centralized display
def get_logo():
    # Try to load logo from assets folder with proper path handling
    logo_path = "assets/images/logo.png"  # Fixed path separator
    
    # Use columns to center the logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if os.path.exists(logo_path):
            return st.image(logo_path, width=200, use_column_width=True)  # Enlarged logo
        else:
            # If logo doesn't exist, use a placeholder SVG
            logo_html = """
            <div style="display: flex; justify-content: center; align-items: center; width: 100%; text-align: center;">
                <svg width="200" height="200" viewBox="0 0 200 200" style="margin: 0 auto;" class="logo-animate">
                    <rect x="50" y="50" width="100" height="100" rx="20" fill="#4da6ff" />
                    <text x="100" y="110" font-family="Arial" font-size="24" fill="white" text-anchor="middle">BizNexus AI</text>
                </svg>
            </div>
            """
            return st.markdown(logo_html, unsafe_allow_html=True)

# Animated typing effect for the tagline
def load_typing_animation():
    typing_js = """
    <script src="https://cdnjs.cloudflare.com/ajax/libs/typed.js/2.0.12/typed.min.js"></script>
    <div style="text-align: center; padding: 20px 0;">
        <span id="typed-tagline" style="font-size: 28px; color: #4da6ff; font-weight: bold;"></span>
    </div>
    <script>
        var typed = new Typed('#typed-tagline', {
            strings: ["Harness AI.", "Predict Success.", "Maximize Growth."],
            typeSpeed: 80,
            backSpeed: 50,
            loop: true,
            backDelay: 1500,
            startDelay: 500
        });
    </script>
    """
    components.html(typing_js, height=100)

# Navigation function
def navigate_to(page_name):
    """
    Global navigation function to navigate between pages
    """
    from streamlit.runtime.scriptrunner import RerunData, RerunException
    from streamlit.source_util import get_pages
    import streamlit as st  # Import streamlit at the function level
    
    def nav_to(page_name: str):
        pages = get_pages("pages")
        for page_hash, page_config in pages.items():
            if page_config["page_name"] == page_name:
                raise RerunException(
                    RerunData(
                        page_script_hash=page_hash,
                        page_name=page_config["page_name"],
                    )
                )
    try:
        nav_to(page_name)
    except Exception as e:
        # Use the imported st instance
        st.error(f"Navigation error: {e}")
def navigate_to(page_path):
    """
    Navigate to a specific page with improved error handling
    
    Args:
        page_path (str): Path to the page, e.g., "pages/02_Authentication.py"
    """
    # Make sure to use the correct path format
    if not page_path.startswith("pages/"):
        page_path = f"pages/{page_path}"
    if not page_path.endswith(".py"):
        page_path = f"{page_path}.py"
    
    try:
        st.switch_page(page_path)
    except Exception as e:
        st.error(f"Navigation error: {str(e)}")
        st.write(f"Tried to navigate to: {page_path}")

# Main function
def main():
    # Load CSS and background
    load_css()
    load_particles_bg()
    st.session_state.authenticated = True
    st.session_state.user_info = {"user_id": "test_user", "email": "test@example.com", "full_name": "Test User", "company_id": "test_company", "company_name": "Test Company", "role": "admin"}
    st.session_state.company_id = "test_company"
    
    # Logo at the top - Just call once
    with st.container():
        st.markdown('<div class="logo-container">', unsafe_allow_html=True)
        get_logo()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tagline with typing animation
    load_typing_animation()
    
    # Hero section
    with st.container():
        st.markdown("""
        <div class="hero-section animate-text">
            <h1 style="font-size: 42px; margin-bottom: 20px; color: #2c3e50;">BizNexus AI</h1>
            <h3 style="color: #5a7baf; margin-bottom: 30px;">The Next-Gen AI-Powered Business Analytics Platform</h3>
            <p style="font-size: 18px; max-width: 800px; margin: 0 auto 30px auto; color: #2c3e50;">
                
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Get Started button - Fixed to correctly navigate to authentication page
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Get Started →", use_container_width=True):
            # Use the navigation function with the correct page name
            navigate_to("pages/02_Authentication.py")
            
            # Alternative direct method if the above doesn't work
            try:
                st.switch_page("pages/02_Authentication.py")
            except:
                try:
                    st.switch_page("02_Authentication")
                except:
                    st.error("Navigation failed. Please check your file structure.")
    # Animated features section
    st.markdown("<h2 style='text-align: center; margin-top: 60px; margin-bottom: 40px; color: #2c3e50;'>Our AI-Powered Features</h2>", unsafe_allow_html=True)
    
    features = [
        {"icon": "💰", "title": "Customer Lifetime Value", "desc": "Identify high-value customers and optimize retention strategies with our AI-powered CLV predictions."},
        {"icon": "🚀", "title": "Churn Prediction", "desc": "Spot at-risk customers before they leave and implement targeted retention campaigns."},
        {"icon": "📈", "title": "Sales Forecasting", "desc": "Accurately predict future sales and revenue with our advanced time-series models."},
        {"icon": "🤖", "title": "AI Business Assistant", "desc": "Get instant answers to your business questions with our conversational AI chatbot."}
    ]
    
    # Create feature boxes with staggered animation
    cols = st.columns(4)
    for i, (col, feature) in enumerate(zip(cols, features)):
        with col:
            st.markdown(f"""
            <div class="feature-box" style="animation-delay: {i * 0.2}s;">
                <div style="font-size: 40px; margin-bottom: 15px;">{feature['icon']}</div>
                <h3 style="color: #2c3e50;">{feature['title']}</h3>
                <p style="color: #2c3e50;">{feature['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # About the app
    st.markdown("<h2 style='text-align: center; margin-top: 70px; margin-bottom: 30px; color: #2c3e50;'>About BizNexus AI</h2>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div style="background-color: rgba(255, 255, 255, 0.7); padding: 30px; border-radius: 15px; text-align: center; animation: fadeIn 1s ease-out; margin-bottom: 50px;">
            <p style="font-size: 18px; line-height: 1.7; color: #2c3e50;">
                🌟 Brief Intro & Key Benefits for Home Page<br>
                🚀 Welcome to BizNexus AI – The Ultimate AI-Driven Business Analytics Platform!<br>
                In today's fast-paced business world, data-driven decisions are the key to staying ahead. BizNexus AI empowers businesses with cutting-edge AI solutions for Customer Lifetime Value (CLV) calculation, Churn Prediction, and Sales Forecasting—all in one automated platform.<br>
                ✨ Why Choose BizNexus AI?<br>
                ✅ AI-Powered Predictions – Automate insights with advanced machine learning models.<br>
                ✅ Smart Customer Analytics – Understand your customers, their value, and potential churn risks.<br>
                ✅ Data-Driven Sales Forecasting – Get accurate predictions to optimize business growth.<br>
                ✅ Interactive Dashboards & KPIs – Visualize key business metrics effortlessly.<br>
                ✅ Automated Reports & Alerts – Get instant insights and downloadable reports.<br>
                ✅ AI Chatbot for Strategy Recommendations – Improve business outcomes with smart suggestions.<br>
                📊 From data to decisions—BizNexus AI makes business intelligence effortless!
            </p>
            
        </div>
        """, unsafe_allow_html=True)
    
    # Team section
    st.markdown("<h2 style='text-align: center; margin-top: 60px; margin-bottom: 40px; color: #2c3e50;'>Meet the Team</h2>", unsafe_allow_html=True)
    
    team_members = [
        {
            "name": "Vibha S Prasad",
            "role": "AIML Undergrad, BNMIT",
            "bio": "Aspiring engineer innovating at the crossroads of AIML and Analytics",
            "email": "vibhaprasad21@gmail.com",
            "linkedin": "https://www.linkedin.com/in/vibha-s-prasad-283146267",
            "github": "https://github.com/Vibhasprasad21",
        },
        {
            "name": "Pragathi M Shetty",
            "role": "AIML Undergrad, BNMIT",
            "bio": "Aspiring AIML Engineer",
            "email": "pragathimshetty2610@gmail.com",
            "linkedin": "https://www.linkedin.com/in/pragathi-m-shetty-239859205",
            "github": "https://github.com/pragathi26",
        }
        
    ]
    
    # Create team member cards
    cols = st.columns(4)
    for i, (col, member) in enumerate(zip(cols, team_members)):
        with col:
            st.markdown(f"""
            <div class="team-card" style="animation-delay: {i * 0.2 + 0.5}s;">
                <div style="width: 100px; height: 100px; background-color: #4da6ff; border-radius: 50%; margin: 0 auto 15px auto; display: flex; justify-content: center; align-items: center; font-size: 36px; color: white;">
                    {member['name'][0]}
                </div>
                <h3 style="color: #2c3e50;">{member['name']}</h3>
                <div style="color: #4da6ff; margin-bottom: 10px;">{member['role']}</div>
                <p style="font-size: 14px; margin-bottom: 20px; color: #2c3e50;">{member['bio']}</p>
                <div>
                    <a href="mailto:{member['email']}" target="_blank" rel="noopener noreferrer" class="social-icon">
                        <i class="fas fa-envelope"></i>
                    </a>
                    <a href="{member['linkedin']}" target="_blank" rel="noopener noreferrer" class="social-icon">
                        <i class="fab fa-linkedin"></i>
                    </a>
                    <a href="{member['github']}" target="_blank" rel="noopener noreferrer" class="social-icon">
                        <i class="fab fa-github"></i>
                    </a>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Load Font Awesome for icons
    st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.0/css/all.min.css">
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <footer style="text-align: center; margin-top: 80px; padding: 20px; color: #5a7baf; font-size: 14px;">
        <p>© 2025 BizNexus AI. All rights reserved.</p>
    </footer>
    """, unsafe_allow_html=True)
    
    # Add a loading animation when transitioning to other pages
    js = """
    <script>
    const links = document.querySelectorAll('a');
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            // Don't add loading animation to external links
            if (!this.href.includes(window.location.host)) return;
            
            // Create overlay
            const overlay = document.createElement('div');
            overlay.style.position = 'fixed';
            overlay.style.top = '0';
            overlay.style.left = '0';
            overlay.style.width = '100%';
            overlay.style.height = '100%';
            overlay.style.backgroundColor = 'rgba(244, 246, 249, 0.8)';
            overlay.style.display = 'flex';
            overlay.style.justifyContent = 'center';
            overlay.style.alignItems = 'center';
            overlay.style.zIndex = '9999';
            
            // Create loading spinner
            const spinner = document.createElement('div');
            spinner.style.border = '5px solid rgba(0, 0, 0, 0.1)';
            spinner.style.borderRadius = '50%';
            spinner.style.borderTop = '5px solid #4da6ff';
            spinner.style.width = '50px';
            spinner.style.height = '50px';
            spinner.style.animation = 'spin 1s linear infinite';
            
            // Add keyframe animation
            const style = document.createElement('style');
            style.innerHTML = `
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            `;
            document.head.appendChild(style);
            
            overlay.appendChild(spinner);
            document.body.appendChild(overlay);
        });
    });
    </script>
    """
    components.html(js)

if __name__ == "__main__":
    # Initialize session state if not already done
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_info" not in st.session_state:
        st.session_state.user_info = None
    if "company_id" not in st.session_state:
        st.session_state.company_id = None
        
    main()