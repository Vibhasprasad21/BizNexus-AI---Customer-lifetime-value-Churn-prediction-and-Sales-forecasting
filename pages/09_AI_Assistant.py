import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
import traceback
import os
# Import authentication decorator and chatbot assistant
from src.auth.session import requires_auth
from src.chatbot.assistant import BizNexusAssistant

class AIAssistantPage:
    def __init__(self):
        """Initialize the enhanced AI Assistant page"""
        # Page configuration
        st.set_page_config(
            page_title="BizNexus AI | Business Analytics Assistant", 
            page_icon="🤖", 
            layout="wide"
        )
        
        # Apply enhanced professional styling
        self._apply_enhanced_styling()
        
        # Initialize the chatbot with business analytics capabilities
        self.assistant = BizNexusAssistant()
        
        # Initialize session state for chat history if it doesn't exist
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Initialize business insights cache
        if 'business_insights_cache' not in st.session_state:
            st.session_state.business_insights_cache = {}
    
    def _apply_enhanced_styling(self):
        """Apply enhanced professional styling with business aesthetics"""
        st.markdown("""
        <style>
        /* Global Theme */
        .main {
            background-color: #f8f9fa;
            color: #2c3e50;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Dashboard Cards */
        .dashboard-card {
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .dashboard-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.1);
        }
        
        /* Card Headers */
        .card-header {
            color: #2c3e50;
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #f0f0f0;
        }
        
        /* Enhanced Chat Interface */
        .chat-container {
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            padding: 1rem 1.5rem;
            height: calc(100vh - 14rem);
            display: flex;
            flex-direction: column;
        }
        .chat-messages {
            flex-grow: 1;
            overflow-y: auto;
            padding-right: 0.5rem;
            margin-bottom: 1rem;
        }
        .chat-input {
            border-top: 1px solid #f0f0f0;
            padding-top: 1rem;
        }
        .chat-message {
            padding: 1rem 1.2rem;
            border-radius: 0.8rem;
            margin-bottom: 1rem;
            max-width: 80%;
            position: relative;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .chat-message.user {
            background: linear-gradient(135deg, #e9f2ff 0%, #ddefff 100%);
            border: 1px solid #d0e6ff;
            border-top-right-radius: 0.2rem;
            align-self: flex-end;
            margin-left: auto;
        }
        .chat-message.assistant {
            background: linear-gradient(135deg, #f8f9fa 0%, #f0f2f5 100%);
            border: 1px solid #e9ecef;
            border-top-left-radius: 0.2rem;
            align-self: flex-start;
        }
        .chat-message .avatar {
            width: 38px;
            height: 38px;
            border-radius: 50%;
            position: absolute;
            bottom: -5px;
        }
        .chat-message.user .avatar {
            right: -15px;
            background-color: #4a90e2;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        .chat-message.assistant .avatar {
            left: -15px;
            background-color: #6c757d;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
        }
        .message-header {
            font-size: 0.8rem;
            color: #7f8c8d;
            margin-bottom: 0.3rem;
            display: flex;
            justify-content: space-between;
        }
        .message-header .time {
            opacity: 0.7;
        }
        .message-content {
            white-space: pre-wrap;
        }
        
        /* Business Analytics Suggestion Chips */
        .suggestion-container {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 1rem 0;
        }
        .suggestion-chip {
            background: linear-gradient(135deg, #f8f9fa 0%, #f0f2f5 100%);
            border: 1px solid #e9ecef;
            border-radius: 18px;
            padding: 0.5rem 1rem;
            font-size: 0.85rem;
            color: #4a90e2;
            cursor: pointer;
            transition: all 0.2s ease;
            white-space: nowrap;
        }
        .suggestion-chip:hover {
            background: linear-gradient(135deg, #e9f2ff 0%, #ddefff 100%);
            border-color: #d0e6ff;
            transform: translateY(-2px);
            box-shadow: 0 2px 5px rgba(0,0,0,0.08);
        }
        .suggestion-chip .icon {
            margin-right: 0.4rem;
            font-size: 0.9rem;
        }
        
        /* Action Button Styling */
        .action-button {
            background: linear-gradient(135deg, #4a90e2 0%, #357bd8 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }
        .action-button:hover {
            background: linear-gradient(135deg, #357bd8 0%, #2c6fc1 100%);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        .action-button .icon {
            font-size: 1.1rem;
        }

        /* Clear History Button */
        .clear-history-button {
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        .clear-history-button:hover {
            background: linear-gradient(135deg, #c0392b 0%, #a93226 100%);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _validate_data_availability(self):
        """
        Validate that required data is available in session state
        
        Returns:
            bool: True if data is available, False otherwise
        """
        required_keys = ['customer_df']
        
        for key in required_keys:
            if key not in st.session_state:
                st.error(f"Missing {key}. Please complete data upload first.")
                return False
        
        return True
    
    def _display_enhanced_chat_interface(self):
        """
        Display an enhanced professional chat interface with business-focused suggestions
        """
        # Chat header
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="card-header">💬 Business Analytics Assistant</h3>', unsafe_allow_html=True)
        
        # Business-focused helper text
        st.markdown("""
        <p style="margin-bottom:15px;">
            I can help you analyze your customer data, provide business insights, and answer questions about your CLV, 
            churn predictions, and sales forecasts. Ask me about specific customers, business metrics, or suggest strategies for improvement.
        </p>
        """, unsafe_allow_html=True)
        
        # Add Clear History button
        if st.button("🗑️ Clear Chat History", key="clear_history", help="Clear the current conversation history"):
            st.session_state.chat_history = []
            st.experimental_rerun()
        
        
        
        
        # Create scrollable container for chat messages
        st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
        
        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user">
                    <div class="message-header">
                        <span>You</span>
                        <span class="time">{message.get("time", datetime.now().strftime('%H:%M'))}</span>
                    </div>
                    <div class="message-content">{message["content"]}</div>
                    <div class="avatar">👤</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant">
                    <div class="message-header">
                        <span>Business AI Assistant</span>
                        <span class="time">{message.get("time", datetime.now().strftime('%H:%M'))}</span>
                    </div>
                    <div class="message-content">{message["content"]}</div>
                    <div class="avatar">🤖</div>
                </div>
                """, unsafe_allow_html=True)
                
                # If there's a visualization attached to this message, display it
                for msg in st.session_state.get('chat_responses', []):
                    if msg.get("text") == message["content"] and msg.get("visualization") is not None:
                        try:
                            with st.container():
                                # Add this style to reduce the gap
                                st.markdown('<div style="margin-top: -15px;">', unsafe_allow_html=True)
                                st.plotly_chart(msg["visualization"], use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error displaying visualization: {str(e)}")
                        
                        # Add business insights if available
                        if len(msg.get("business_insights", {})) > 0:
                            with st.expander("View Business Insights Details"):
                                for key, value in msg["business_insights"].items():
                                    if isinstance(value, dict):
                                        st.json(value)
                                    elif isinstance(value, (int, float)):
                                        st.metric(key, value)
                                    else:
                                        st.write(f"**{key}:** {value}")
                        
                        # Add action items if available
                        if len(msg.get("action_items", [])) > 0:
                            with st.expander("Recommended Business Actions"):
                                for item in msg["action_items"]:
                                    if isinstance(item, dict):
                                        priority_color = {
                                            "High": "#e74c3c",
                                            "Medium": "#f39c12",
                                            "Low": "#27ae60"
                                        }.get(item.get("priority", "Medium"), "#7f8c8d")
                                        
                                        st.markdown(f"""
                                        <div style="padding:10px 0; border-bottom:1px solid #f0f0f0;">
                                            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                                                <span style="font-weight:600;">{item.get("description", "")}</span>
                                                <span style="background-color:rgba({priority_color},0.1); color:{priority_color}; padding:2px 8px; border-radius:10px; font-size:0.8rem;">{item.get("priority", "Medium")}</span>
                                            </div>
                                            <div style="color:#7f8c8d; font-size:0.9rem;">{item.get("expected_impact", "")}</div>
                                            <div style="color:#6c757d; font-size:0.8rem; margin-top:5px;">Owner: {item.get("owner", "")}</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"- {item}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input
        st.markdown('<div class="chat-input">', unsafe_allow_html=True)
        
        chat_cols = st.columns([5, 1])
        
        with chat_cols[0]:
            chat_input = st.text_area("Ask me about your business metrics, specific customers, or strategies", 
                              key="chat_input", height=80, max_chars=500,
                              placeholder="Example: What's the CLV for customer ID 1001? or Show me high-value customers at risk of churning")
        
        with chat_cols[1]:
            st.markdown("<br>", unsafe_allow_html=True)  # Add vertical spacing
            send_button = st.button("Send", type="primary", use_container_width=True)
        
        if send_button and chat_input:
            self._process_user_input(chat_input)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)  # Close the dashboard card
    
    def _process_user_input(self, user_input):
        """
        Process user input and generate business-focused response
        
        Args:
            user_input (str): User's message
        """
        # Add user message to chat history with timestamp
        timestamp = datetime.now().strftime('%H:%M')
        st.session_state.chat_history.append({"role": "user", "content": user_input, "time": timestamp})
        
        # Initialize chat_responses if it doesn't exist
        if 'chat_responses' not in st.session_state:
            st.session_state.chat_responses = []
        
        try:
            # Generate response from the assistant
            response = self.assistant.generate_response(user_input, st.session_state)
            
            # Add assistant response to chat history with timestamp
            st.session_state.chat_history.append({"role": "assistant", "content": response["text"], "time": timestamp})
            
            # Save the full response (including visualizations) to a separate list
            st.session_state.chat_responses.append(response)
        except Exception as e:
            error_message = f"I encountered an error while processing your request: {str(e)}\n\nPlease try a different question or rephrase your query."
            st.session_state.chat_history.append({"role": "assistant", "content": error_message, "time": timestamp})
            # Log the full error for debugging
            print(f"Error processing input: {traceback.format_exc()}")
        
        # Rerun to update the UI
        st.experimental_rerun()
    
    def render(self):
        """
        Main rendering method for enhanced AI Assistant page
        """
        st.title("🤖 BizNexus Business Analytics Assistant")
        
        # Validate data availability
        if not self._validate_data_availability():
            return
        
        # Display enhanced chat interface (full width)
        self._display_enhanced_chat_interface()


# Add main function with authentication decorator
@requires_auth
def main():
    """
    Main function to initialize and render the AI Assistant page
    """
    # Create an instance of the AIAssistantPage class
    ai_assistant_page = AIAssistantPage()
    
    # Render the page
    ai_assistant_page.render()

# Ensure the script can be run directly
if __name__ == "__main__":
    main()