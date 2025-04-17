import streamlit as st
import time
from src.auth.firebase_auth import login_user, register_user, is_valid_email
import streamlit.components.v1 as components
import os
from dotenv import load_dotenv
load_dotenv()
# Custom CSS for styling
def load_css():
    st.markdown("""
    <style>
        .main {
            background-color: #f4f6f9;  /* Soft pastel blue-gray background */
            color: #2c3e50;  /* Dark text for readability */
        }
        
        .upload-container {
            background-color: #ffffff;  /* Clean white background */
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        
        .upload-container:hover {
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
        }
        
        .stButton > button {
            background-color: #6a89cc !important;  /* Soft pastel blue */
            color: white !important;
            border: none !important;
            padding: 12px 24px !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton > button:hover {
            background-color: #5a7baf !important;  /* Slightly darker shade */
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
        }
        
        .upload-area {
            border: 2px dashed #6a89cc;
            border-radius: 12px;
            padding: 40px 20px;
            text-align: center;
            margin-bottom: 20px;
            background-color: #f0f4f8;  /* Light pastel background */
            transition: all 0.3s ease;
        }
        
        .upload-area:hover {
            border-color: #5a7baf;
            background-color: #e6eaf4;  /* Slightly different pastel shade */
        }
        
        /* Center and enlarge the logo */
        .logo-container {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            margin: 0 auto !important;
            text-align: center !important;
            margin-bottom: 20px;
            width: 100% !important;
        }
        
        .logo-container img {
            width: 180px !important;  /* Enlarge from 120px to 180px */
            margin: 0 auto !important;
            display: block !important;
        }
    </style>
    """, unsafe_allow_html=True)


# Load animated background
def load_particles_bg():
    particles_js = """
    <div id="tsparticles"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/tsparticles/2.9.3/tsparticles.bundle.min.js"></script>
    <script>
        tsParticles.load("tsparticles", {
            fpsLimit: 60,
            particles: {
                number: {
                    value: 50,
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
                    speed: 0.8,
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
                        particles_nb: 3
                    }
                }
            },
            retina_detect: true
        });
    </script>
    """
    components.html(particles_js, height=0)

# Get logo
def get_logo():
    # Try to load logo from assets folder
    logo_path = "assets/images/logo.png"
    if os.path.exists(logo_path):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            return st.image(logo_path, width=180, use_column_width=True)  # Increased from 120 to 180
    else:
        # If logo doesn't exist, use a placeholder SVG
        logo_html = """
        <div style="display: flex; justify-content: center; align-items: center; width: 100%; text-align: center;">
            <svg width="180" height="180" viewBox="0 0 200 200" style="margin: 0 auto;">
                <rect x="50" y="50" width="100" height="100" rx="20" fill="#4da6ff" />
                <text x="100" y="110" font-family="Arial" font-size="24" fill="white" text-anchor="middle">BizNexus AI</text>
            </svg>
        </div>
        """
        return st.markdown(logo_html, unsafe_allow_html=True)

# Main function
def main():
    # Check if already authenticated, redirect to upload page
    if st.session_state.get('authenticated', False):
        # Use direct navigation instead
        try:
            st.switch_page("pages/03_Upload.py")
        except Exception as e:
            st.error(f"Navigation error: {str(e)}")
    
    # Load CSS and particles background
    load_css()
    load_particles_bg()
    
    # Logo at the top
    with st.container():
        st.markdown('<div class="logo-container">', unsafe_allow_html=True)
        get_logo()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Page title
    st.markdown('<h1 style="text-align: center; margin-bottom: 30px;">Welcome to BizNexus AI</h1>', unsafe_allow_html=True)
    
    # Authentication container
    with st.container():
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        
        # Create tabs for login and registration
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.markdown('<h2 class="form-header">Business Analyst Login</h2>', unsafe_allow_html=True)
            
            email = st.text_input("Email", key="login_email", placeholder="Enter your business email")
            password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                login_button = st.button("Login", use_container_width=True)
            
            if login_button:
                if not email or not password:
                    st.error("Please enter both email and password")
                else:
                    with st.spinner("Logging in..."):
                        result = login_user(email, password)
                        
                        if result["success"]:
                            # Store user information in session state
                            st.session_state.authenticated = True
                            st.session_state.user_info = {
                                "user_id": result["user_id"],
                                "email": result["email"],
                                "full_name": result["full_name"],
                                "company_id": result["company_id"],
                                "company_name": result["company_name"],
                                "role": result["role"]
                            }
                            st.session_state.company_id = result["company_id"]
                            
                            st.success("Login successful!")
                            time.sleep(1)  # Brief pause to show the success message
                            
                            # Direct navigation approach
                            try:
                                st.switch_page("pages/03_Upload.py")
                            except Exception as e:
                                st.error(f"Navigation error: {str(e)}")
                                st.info("Please manually navigate to the Upload page.")
                        else:
                            st.error(f"Login failed: {result['error']}")
        
        with tab2:
            st.markdown('<h2 class="form-header">Create Your BizNexus Account</h2>', unsafe_allow_html=True)
            
            full_name = st.text_input("Full Name", key="reg_name", placeholder="Enter your full name")
            email = st.text_input("Business Email", key="reg_email", placeholder="Enter your business email")
            company = st.text_input("Company Name", key="reg_company", placeholder="Enter your company name")
            password = st.text_input("Password", type="password", key="reg_password", 
                                     placeholder="Create a strong password (min. 8 characters)")
            confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm", 
                                            placeholder="Confirm your password")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                register_button = st.button("Create Account", use_container_width=True)
            
            if register_button:
                # Validate form inputs
                if not full_name or not email or not company or not password or not confirm_password:
                    st.error("Please fill in all fields")
                elif not is_valid_email(email):
                    st.error("Please enter a valid email address")
                elif len(password) < 8:
                    st.error("Password must be at least 8 characters long")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    with st.spinner("Creating your account..."):
                        result = register_user(email, password, full_name, company)
                        
                        if result["success"]:
                            st.success("Account created successfully! Please log in.")
                            time.sleep(1.5)
                            # Switch to login tab
                            st.experimental_rerun()
                        else:
                            st.error(f"Registration failed: {result['error']}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <footer style="text-align: center; margin-top: 50px; padding: 20px; color: #8b9dc3; font-size: 14px;">
        <p>© 2025 BizNexus AI. All rights reserved.</p>
    </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Initialize session state if not already done
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_info" not in st.session_state:
        st.session_state.user_info = None
    if "company_id" not in st.session_state:
        st.session_state.company_id = None
    
    main()