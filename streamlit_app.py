import streamlit as st
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv
from groq import Groq

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Work Delegation System",
    page_icon="📋",
    layout="wide"
)

load_dotenv()

# Initialize Groq client for AI features
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.warning("GROQ_API_KEY not found. AI features will be limited.")
    client = None
else:
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        st.warning(f"Error initializing AI client: {str(e)}")
        client = None

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_role' not in st.session_state:
    st.session_state.user_role = None

def logout():
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.user_role = None
    st.rerun()

def analyze_user_performance(username, performance_data):
    """AI-powered user performance analysis"""
    if not client:
        return "AI analysis is not available. Please check GROQ API configuration."
    
    try:
        messages = [
            {
                "role": "system",
                "content": """You are a performance analyst specializing in individual performance evaluation.
                Analyze the user's performance data and provide:
                1. Overall performance assessment
                2. Key strengths and achievements
                3. Areas needing improvement
                4. Specific actionable recommendations
                5. Performance trends and patterns
                Be specific and data-driven in your analysis."""
            },
            {
                "role": "user",
                "content": f"Analyze performance data for user {username}: {performance_data}"
            }
        ]
        
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"AI Analysis Error: {str(e)}")
        return "Unable to analyze user performance at this time. Please try again later."

def analyze_work_item(work_data):
    """AI-powered work analysis"""
    if not client:
        return "AI analysis is not available. Please check GROQ API configuration."
    
    try:
        messages = [
            {
                "role": "system",
                "content": """You are a work analysis expert. For the given work item:
                1. Assess complexity and requirements
                2. Identify potential challenges
                3. Suggest optimal resource allocation
                4. Provide risk assessment
                5. Recommend best practices
                Be practical and specific in your recommendations."""
            },
            {
                "role": "user",
                "content": f"Analyze this work item: {work_data}"
            }
        ]
        
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"AI Analysis Error: {str(e)}")
        return "Unable to analyze work item at this time. Please try again later."

def get_improvement_recommendations(user_data, work_history):
    """AI-powered improvement recommendations"""
    if not client:
        return "AI recommendations are not available. Please check GROQ API configuration."
    
    try:
        messages = [
            {
                "role": "system",
                "content": """You are an AI coach specializing in professional development.
                Based on the user's data and work history, provide:
                1. Personalized improvement strategies
                2. Skill development recommendations
                3. Productivity enhancement tips
                4. Time management suggestions
                5. Career growth opportunities
                Make recommendations specific and actionable."""
            },
            {
                "role": "user",
                "content": f"Generate improvement recommendations based on:\nUser Data: {user_data}\nWork History: {work_history}"
            }
        ]
        
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"AI Recommendations Error: {str(e)}")
        return "Unable to generate recommendations at this time. Please try again later."

def get_team_analytics(team_data):
    """Generate team performance analytics using AI"""
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a team performance analyst. Analyze team data and provide actionable insights."
            },
            {
                "role": "user",
                "content": f"Analyze this team's performance data and provide key insights: {team_data}"
            }
        ]
        
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error in team analysis: {str(e)}"

def get_team_specific_recommendations(team_data, task_description):
    """Get AI recommendations specific to the team's context"""
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a team optimization expert. Consider team context when making recommendations."
            },
            {
                "role": "user",
                "content": f"Given this team context: {team_data}\n\nProvide recommendations for task: {task_description}"
            }
        ]
        
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error in team recommendations: {str(e)}"

def show_team_chat():
    st.subheader("Team Chat")
    
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    # Display chat messages
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(f"{msg['sender']}: {msg['content']}")
            st.caption(msg["timestamp"])
    
    # Chat input
    message = st.chat_input("Type your message...")
    if message:
        try:
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            response = requests.post(
                f"{os.getenv('API_URL')}/api/chat/send",
                headers=headers,
                json={"content": message}
            )
            if response.ok:
                new_message = response.json()
                st.session_state.chat_messages.append(new_message)
                st.rerun()
        except Exception as e:
            st.error(f"Error sending message: {str(e)}")

def check_delegation_rules(task_data, assignee_data):
    """Check if task delegation follows team rules"""
    try:
        messages = [
            {
                "role": "system",
                "content": """You are a work delegation expert. Evaluate if task assignment follows these rules:
                1. Task complexity matches assignee's experience level
                2. Current workload is balanced
                3. Skills match task requirements
                4. Priority levels are properly handled
                Return a JSON with: {"allowed": boolean, "reason": string}"""
            },
            {
                "role": "user",
                "content": f"Evaluate this task delegation:\nTask: {task_data}\nAssignee: {assignee_data}"
            }
        ]
        
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error in delegation rules check: {str(e)}"

def show_team_management():
    st.subheader("Team Management")
    
    if st.session_state.user_role != "team_leader":
        st.info("Only team leaders can access team management features.")
        return
    
    # Get team members based on current user
    team_members = st.session_state.team
    
    # Team Analytics Tab
    tabs = st.tabs(["📈 Analytics", "👥 Members", "🎯 KPIs", "💬 Chat"])
    analytics_tab, members_tab, kpi_tab, chat_tab = tabs
    
    with analytics_tab:
        st.subheader("Team Performance Analytics")
        
        # Display team metrics
        cols = st.columns(3)
        with cols[0]:
            st.metric("Team Efficiency", f"{st.session_state.user_performance['team_efficiency']}%")
        with cols[1]:
            st.metric("Projects Completed", st.session_state.user_performance['projects_completed'])
        with cols[2]:
            st.metric("Team Satisfaction", f"{st.session_state.user_performance['team_satisfaction']}/5")
        
        # Get AI insights
        with st.expander("🤖 AI Team Analysis"):
            team_data = {
                "metrics": st.session_state.user_performance,
                "team_size": len(team_members),
                "current_projects": 5  # Example value
            }
            analysis = get_team_analytics(team_data)
            st.write(analysis)
    
    with members_tab:
        st.subheader("Team Members")
        for member in team_members:
            with st.expander(f"📊 {member}"):
                if member in TEST_PERFORMANCE:
                    perf = TEST_PERFORMANCE[member]
                    cols = st.columns(2)
                    with cols[0]:
                        st.metric("Tasks Completed", perf['tasks_completed'])
                        st.metric("Completion Rate", f"{perf['completion_rate']}%")
                    with cols[1]:
                        st.metric("Quality Score", f"{perf['quality_score']}/5")
                        st.metric("Response Time", perf['response_time'])
                    
                    # Add work assignment button for each member
                    if st.button(f"Assign Work to {member}", key=f"assign_{member}"):
                        st.session_state.assigning_to = member
                        st.session_state.show_assignment_form = True
                else:
                    st.warning(f"Performance data not available for {member}")
    
    with kpi_tab:
        st.subheader("Team KPIs")
        for member in team_members:
            if member in TEST_PERFORMANCE:
                with st.expander(f"📈 {member}'s KPIs"):
                    perf = TEST_PERFORMANCE[member]
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Team Contribution", f"{perf['team_contribution']}/5")
                    with cols[1]:
                        st.metric("Quality Score", f"{perf['quality_score']}/5")
                    with cols[2]:
                        st.metric("Response Time", perf['response_time'])
    
    with chat_tab:
        show_team_chat()
    
    # Work Assignment Form
    if st.session_state.get('show_assignment_form', False):
        st.subheader(f"Assign Work to {st.session_state.assigning_to}")
        with st.form("work_assignment_form"):
            title = st.text_input("Task Title")
            description = st.text_area("Task Description")
            priority = st.selectbox("Priority", ["low", "medium", "high"])
            estimated_hours = st.number_input("Estimated Hours", min_value=1, max_value=100)
            
            cols = st.columns(2)
            with cols[0]:
                get_prediction = st.checkbox("Get AI Timeline Prediction")
            with cols[1]:
                get_recommendations = st.checkbox("Get Team-Specific Recommendations")
            
            if st.form_submit_button("Assign Task"):
                if title and description:
                    # Here you would normally make an API call to assign the task
                    st.success(f"Task assigned to {st.session_state.assigning_to}!")
                    
                    if get_prediction:
                        with st.spinner("Predicting completion timeline..."):
                            prediction = predict_completion(description)
                            st.info(f"🕒 AI Timeline Prediction:\n{prediction}")
                    
                    if get_recommendations:
                        with st.spinner("Generating team-specific recommendations..."):
                            member_data = TEST_PERFORMANCE.get(st.session_state.assigning_to, {})
                            recommendations = get_team_specific_recommendations(member_data, description)
                            st.success(f"💡 Team-Specific Recommendations:\n{recommendations}")
                    
                    # Clear the assignment form
                    st.session_state.show_assignment_form = False
                    st.rerun()
                else:
                    st.error("Please fill in all required fields")

def show_dashboard():
    st.title("Work Delegation Dashboard")
    
    # Sidebar for actions and AI analysis
    with st.sidebar:
        st.button("Logout", on_click=logout)
        st.subheader("AI Assistant")
        
        # Performance Analysis
        st.subheader("📊 Performance Analysis")
        if st.button("Analyze My Performance"):
            with st.spinner("Analyzing performance..."):
                insights = analyze_user_performance(
                    st.session_state.username,
                    {
                        "historical": HISTORICAL_PERFORMANCE.get(st.session_state.username, []),
                        "current": TEST_PERFORMANCE.get(st.session_state.username, {})
                    }
                )
                st.info(insights)
        
        # Work Analysis
        st.subheader("🎯 Work Analysis")
        work_query = st.text_area("Describe work to analyze:")
        if st.button("Analyze Work"):
            with st.spinner("Analyzing work..."):
                analysis = analyze_work_item({"description": work_query})
                st.info(analysis)
        
        # Improvement Recommendations
        st.subheader("💡 Get Recommendations")
        if st.button("Get Improvement Suggestions"):
            with st.spinner("Generating recommendations..."):
                recommendations = get_improvement_recommendations(
                    TEST_PERFORMANCE.get(st.session_state.username, {}),
                    HISTORICAL_PERFORMANCE.get(st.session_state.username, [])
                )
                st.success(recommendations)
    
    # Main dashboard content
    tabs = st.tabs(["📊 Work Items", "📈 Analytics", "👥 Team Management"])
    work_tab, analytics_tab, team_tab = tabs
    
    with work_tab:
        # Show metrics
        cols = st.columns(3)
        with cols[0]:
            st.metric("Total Tasks", "5")
        with cols[1]:
            st.metric("Completed", "2")
        with cols[2]:
            st.metric("Pending", "3")
        
        # Work Items Table
        st.subheader("Work Items")
        dummy_items = [
            {"title": "Task 1", "description": "Sample task 1", "status": "pending"},
            {"title": "Task 2", "description": "Sample task 2", "status": "completed"},
            {"title": "Task 3", "description": "Sample task 3", "status": "in_progress"}
        ]
        st.table(dummy_items)
        
        # Add New Work Item form
        st.subheader("Add New Work Item")
        with st.form("new_work_item"):
            title = st.text_input("Title")
            description = st.text_area("Description")
            priority = st.selectbox("Priority", ["low", "medium", "high"])
            estimated_hours = st.number_input("Estimated Hours", min_value=1, max_value=100)
            
            # Show assignee selection only for team leaders
            assignee = None
            if st.session_state.user_role == "team_leader":
                assignee = st.selectbox("Assign to", ["Self"] + st.session_state.team)
            
            cols = st.columns(2)
            with cols[0]:
                get_prediction = st.checkbox("Get AI Timeline Prediction")
            with cols[1]:
                get_recommendations = st.checkbox("Get Team-Specific Recommendations")
            
            if st.form_submit_button("Add Item"):
                if title and description:
                    # Check assignment permissions
                    target_assignee = assignee if assignee and assignee != "Self" else st.session_state.username
                    if can_assign_work(st.session_state.user_role, st.session_state.team, target_assignee):
                        st.success(f"Work item assigned to {target_assignee}!")
                        
                        if get_prediction:
                            with st.spinner("Predicting completion timeline..."):
                                prediction = predict_completion(description)
                                st.info(f"🕒 AI Timeline Prediction:\n{prediction}")
                        
                        if get_recommendations:
                            with st.spinner("Generating recommendations..."):
                                if target_assignee in TEST_PERFORMANCE:
                                    member_data = TEST_PERFORMANCE[target_assignee]
                                    recommendations = get_team_specific_recommendations(member_data, description)
                                    st.success(f"💡 Team-Specific Recommendations:\n{recommendations}")
                                else:
                                    recommendations = get_ai_recommendations(description)
                                    st.success(f"💡 Recommendations:\n{recommendations}")
                        
                        st.rerun()
                    else:
                        st.error("You don't have permission to assign work to this team member")
                else:
                    st.error("Please fill in all required fields")
    
    with analytics_tab:
        show_performance_analytics()
    
    with team_tab:
        if st.session_state.user_role == "team_leader":
            show_team_management()
        else:
            st.info("Team management features are only available to team leaders")

def predict_completion(description):
    """AI-powered completion time prediction"""
    if not client:
        return "AI prediction is not available. Please check GROQ API configuration."
    
    try:
        messages = [
            {
                "role": "system",
                "content": """You are a project timeline analyst. For the given task:
                1. Estimate completion timeline
                2. Identify potential delays
                3. Suggest timeline optimization
                4. Provide confidence level
                Be specific and practical in your analysis."""
            },
            {
                "role": "user",
                "content": f"Predict completion timeline for: {description}"
            }
        ]
        
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"AI Prediction Error: {str(e)}")
        return "Unable to predict completion time at this time. Please try again later."

def get_ai_recommendations(description):
    """AI-powered task recommendations"""
    if not client:
        return "AI recommendations are not available. Please check GROQ API configuration."
    
    try:
        messages = [
            {
                "role": "system",
                "content": """You are a task optimization expert. For the given task:
                1. Suggest best practices
                2. Identify optimization opportunities
                3. Provide resource recommendations
                4. List potential improvements
                Be practical and specific in your suggestions."""
            },
            {
                "role": "user",
                "content": f"Provide recommendations for: {description}"
            }
        ]
        
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"AI Recommendations Error: {str(e)}")
        return "Unable to generate recommendations at this time. Please try again later."

# Performance Metrics
TEST_PERFORMANCE = {
    "vineeth": {
        "tasks_completed": 12,
        "completion_rate": 85,
        "quality_score": 4.2,
        "response_time": "2 hours",
        "team_contribution": 4.5
    },
    "nag": {
        "tasks_completed": 15,
        "completion_rate": 90,
        "quality_score": 4.4,
        "response_time": "1.5 hours",
        "team_contribution": 4.7
    },
    "ramya": {
        "tasks_completed": 10,
        "completion_rate": 82,
        "quality_score": 4.1,
        "response_time": "2.5 hours",
        "team_contribution": 4.3
    },
    "sruthi": {
        "tasks_completed": 14,
        "completion_rate": 88,
        "quality_score": 4.3,
        "response_time": "1.8 hours",
        "team_contribution": 4.6
    },
    "venkat": {
        "tasks_completed": 11,
        "completion_rate": 84,
        "quality_score": 4.0,
        "response_time": "2.2 hours",
        "team_contribution": 4.4
    }
}

# Historical Performance Data (Last Week)
HISTORICAL_PERFORMANCE = {
    "vineeth": [
        {"date": "2024-03-20", "tasks_completed": 3, "quality_score": 4.3, "response_time": "1.8 hours"},
        {"date": "2024-03-19", "tasks_completed": 2, "quality_score": 4.1, "response_time": "2.1 hours"},
        {"date": "2024-03-18", "tasks_completed": 4, "quality_score": 4.4, "response_time": "1.9 hours"},
        {"date": "2024-03-17", "tasks_completed": 2, "quality_score": 4.0, "response_time": "2.2 hours"},
        {"date": "2024-03-16", "tasks_completed": 3, "quality_score": 4.2, "response_time": "2.0 hours"}
    ],
    "nag": [
        {"date": "2024-03-20", "tasks_completed": 4, "quality_score": 4.5, "response_time": "1.4 hours"},
        {"date": "2024-03-19", "tasks_completed": 3, "quality_score": 4.3, "response_time": "1.6 hours"},
        {"date": "2024-03-18", "tasks_completed": 4, "quality_score": 4.6, "response_time": "1.3 hours"},
        {"date": "2024-03-17", "tasks_completed": 3, "quality_score": 4.4, "response_time": "1.5 hours"},
        {"date": "2024-03-16", "tasks_completed": 4, "quality_score": 4.5, "response_time": "1.4 hours"}
    ],
    "ramya": [
        {"date": "2024-03-20", "tasks_completed": 2, "quality_score": 4.0, "response_time": "2.4 hours"},
        {"date": "2024-03-19", "tasks_completed": 3, "quality_score": 4.2, "response_time": "2.3 hours"},
        {"date": "2024-03-18", "tasks_completed": 2, "quality_score": 4.1, "response_time": "2.5 hours"},
        {"date": "2024-03-17", "tasks_completed": 2, "quality_score": 4.0, "response_time": "2.6 hours"},
        {"date": "2024-03-16", "tasks_completed": 3, "quality_score": 4.2, "response_time": "2.4 hours"}
    ],
    "sruthi": [
        {"date": "2024-03-20", "tasks_completed": 3, "quality_score": 4.4, "response_time": "1.7 hours"},
        {"date": "2024-03-19", "tasks_completed": 4, "quality_score": 4.3, "response_time": "1.8 hours"},
        {"date": "2024-03-18", "tasks_completed": 3, "quality_score": 4.2, "response_time": "1.9 hours"},
        {"date": "2024-03-17", "tasks_completed": 3, "quality_score": 4.4, "response_time": "1.7 hours"},
        {"date": "2024-03-16", "tasks_completed": 4, "quality_score": 4.3, "response_time": "1.8 hours"}
    ],
    "venkat": [
        {"date": "2024-03-20", "tasks_completed": 2, "quality_score": 4.1, "response_time": "2.1 hours"},
        {"date": "2024-03-19", "tasks_completed": 3, "quality_score": 4.0, "response_time": "2.2 hours"},
        {"date": "2024-03-18", "tasks_completed": 2, "quality_score": 3.9, "response_time": "2.3 hours"},
        {"date": "2024-03-17", "tasks_completed": 3, "quality_score": 4.1, "response_time": "2.1 hours"},
        {"date": "2024-03-16", "tasks_completed": 2, "quality_score": 4.0, "response_time": "2.2 hours"}
    ]
}

def get_performance_insights(username, historical_data):
    """Get AI insights on performance trends"""
    if not client:
        return "AI analysis is not available. Please check GROQ API configuration."
    
    try:
        messages = [
            {
                "role": "system",
                "content": """You are a performance analyst. Analyze the given historical performance data and provide:
                1. Key trends in performance
                2. Areas of improvement
                3. Specific recommendations
                4. Comparison with team averages
                Be specific and actionable in your recommendations."""
            },
            {
                "role": "user",
                "content": f"Analyze this performance data for {username}: {historical_data}"
            }
        ]
        
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"AI Analysis Error: {str(e)}")
        return "Unable to analyze performance at this time. Please try again later."

def can_assign_work(assigner_role, assigner_team, assignee):
    """Check if user can assign work to the specified team member"""
    if assigner_role == "team_leader":
        return assignee in assigner_team
    elif assigner_role == "team_member":
        return assigner_team == assignee  # Can only assign to self
    return False

def show_performance_analytics():
    st.subheader("Performance Analytics")
    
    # Show different views based on user role
    if st.session_state.user_role == "team_leader":
        # Leaders can see all team members' analytics
        member = st.selectbox("Select Team Member", ["All Team"] + st.session_state.team)
        if member == "All Team":
            show_team_analytics()
        else:
            show_member_analytics(member)
    else:
        # Team members can only see their own analytics
        show_member_analytics(st.session_state.username)

def show_member_analytics(username):
    """Show detailed analytics for a specific team member"""
    if username in HISTORICAL_PERFORMANCE:
        data = HISTORICAL_PERFORMANCE[username]
        current_perf = TEST_PERFORMANCE.get(username, {})
        
        # Show metrics over time
        st.subheader(f"Performance Trends for {username}")
        
        # Convert data for charts
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Tasks Completed")
            st.line_chart(df.set_index('date')['tasks_completed'])
        
        with col2:
            st.subheader("Quality Score")
            st.line_chart(df.set_index('date')['quality_score'])
        
        # AI Analysis Tabs
        analysis_tabs = st.tabs(["🎯 Performance Analysis", "📊 Work Analysis", "💡 Recommendations"])
        
        with analysis_tabs[0]:
            with st.spinner("Analyzing performance..."):
                performance_insights = analyze_user_performance(username, {
                    "historical": data,
                    "current": current_perf
                })
                st.write(performance_insights)
        
        with analysis_tabs[1]:
            with st.spinner("Analyzing work patterns..."):
                work_insights = analyze_work_item({
                    "user": username,
                    "performance": current_perf,
                    "history": data
                })
                st.write(work_insights)
        
        with analysis_tabs[2]:
            with st.spinner("Generating recommendations..."):
                recommendations = get_improvement_recommendations(
                    current_perf,
                    data
                )
                st.write(recommendations)
    else:
        st.warning(f"No historical data available for {username}")

def show_team_analytics():
    """Show analytics for the entire team"""
    st.subheader("Team Overview")
    
    # Calculate team averages
    team_data = {}
    for member in st.session_state.team:
        if member in HISTORICAL_PERFORMANCE:
            data = HISTORICAL_PERFORMANCE[member]
            df = pd.DataFrame(data)
            team_data[member] = {
                "avg_tasks": df['tasks_completed'].mean(),
                "avg_quality": df['quality_score'].mean(),
                "total_tasks": df['tasks_completed'].sum()
            }
    
    # Show team comparison
    if team_data:
        comparison_df = pd.DataFrame(team_data).T
        st.bar_chart(comparison_df['avg_tasks'])
        st.bar_chart(comparison_df['avg_quality'])
        
        # Get AI Team Insights
        with st.expander("🤖 AI Team Analysis"):
            team_insights = get_team_analytics(team_data)
            st.write(team_insights)

# Main app logic
def main():
    # User Switcher in sidebar
    st.sidebar.title("User View Switcher")
    
    # Define hierarchy with performance data
    hierarchy = {
        "Top Level": {
            "vamsi": {
                "role": "team_leader",
                "team": ["arjun", "vineeth", "nag", "venkat"],
                "performance": {
                    "team_efficiency": 88,
                    "projects_completed": 15,
                    "team_satisfaction": 4.5
                }
            }
        },
        "Middle Level": {
            "arjun": {
                "role": "team_leader",
                "team": ["ramya", "sruthi"],
                "performance": {
                    "team_efficiency": 85,
                    "projects_completed": 8,
                    "team_satisfaction": 4.3
                }
            }
        },
        "Team Members": {
            "vineeth": {
                "role": "team_member",
                "leader": "vamsi",
                "performance": TEST_PERFORMANCE["vineeth"]
            },
            "nag": {
                "role": "team_member",
                "leader": "vamsi",
                "performance": TEST_PERFORMANCE["nag"]
            },
            "ramya": {
                "role": "team_member",
                "leader": "arjun",
                "performance": TEST_PERFORMANCE["ramya"]
            },
            "sruthi": {
                "role": "team_member",
                "leader": "arjun",
                "performance": TEST_PERFORMANCE["sruthi"]
            },
            "venkat": {
                "role": "team_member",
                "leader": "vamsi",
                "performance": TEST_PERFORMANCE["venkat"]
            }
        }
    }
    
    # Allow switching between users
    level = st.sidebar.selectbox("Select Level", list(hierarchy.keys()))
    users_in_level = list(hierarchy[level].keys())
    selected_user = st.sidebar.selectbox("Select User", users_in_level)
    
    if st.sidebar.button("Switch User View"):
        user_data = hierarchy[level][selected_user]
        st.session_state.authenticated = True
        st.session_state.username = selected_user
        st.session_state.user_role = user_data["role"]
        st.session_state.user_performance = user_data["performance"]
        if "team" in user_data:
            st.session_state.team = user_data["team"]
        if "leader" in user_data:
            st.session_state.team_leader = user_data["leader"]
        st.rerun()
    
    # Show current user view
    if st.session_state.authenticated:
        st.sidebar.success(f"Current View: {st.session_state.username} ({st.session_state.user_role})")
        show_dashboard()
    else:
        st.warning("Please select a user and click 'Switch User View'")

if __name__ == "__main__":
    main() 