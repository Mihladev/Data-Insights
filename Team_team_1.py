pip install -r requirements.txt
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="Job Market Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"  
)

# Loading data
@st.cache_data
def load_data():
    df = pd.read_csv("Team_1.csv")
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]  
    return df

jobs = load_data()

# Styling
st.markdown("""
    <style>
        .css-18e3th9 {
            padding: 1rem 2rem;
        }
        .main { 
            background-color: #f4f4f4;
        }
        .st-bb {
            margin-top: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

# Overview 
st.title("📊 Job Market Analysis")
st.write("Explore insights from the job market based on the dataset of job descriptions, required skills, and salaries.")

# Dataset preview
st.write("### Dataset Preview")
st.dataframe(jobs.head())


st.write("### Dataset Information")
col1, col2 = st.columns(2)
col1.metric("Total Rows", jobs.shape[0])
col2.metric("Total Columns", jobs.shape[1])


st.write("#### Insights")
st.write(f"""
- The dataset contains {jobs.shape[0]} job descriptions and {jobs.shape[1]} columns of information.
- Key columns include 'Job Description', 'Skills Required', 'Salary Range', and 'Experience Level', which provide comprehensive data for analysis.
""")

st.write("### Columns in the Dataset")
st.write(", ".join(jobs.columns))

# Experience Level Filter 
experience_levels = jobs['Experience_Level'].unique().tolist()
selected_experience = st.selectbox("Select Experience Level", ["All"] + experience_levels)


if selected_experience != "All":
    filtered_jobs = jobs[jobs['Experience_Level'] == selected_experience]
else:
    filtered_jobs = jobs


all_tokens = [word for tokens in filtered_jobs["Tokenized_Job_Description"] for word in eval(tokens)]
token_freq = Counter(all_tokens)
most_common_words = token_freq.most_common(20)
words, counts = zip(*most_common_words)

skills_list = [skill.strip().lower() for skills in filtered_jobs["Skills_Required"] for skill in skills.split(",")]
skill_freq = Counter(skills_list)
top_skills = skill_freq.most_common(10)
skills, skill_counts = zip(*top_skills)

sentiment_counts = filtered_jobs["Sentiment_Label"].value_counts()


fig, axs = plt.subplots(2, 2, figsize=(18, 12))

# Top 20 Most Common Words Plot
axs[0, 0].bar(words, counts, color='mediumslateblue')
axs[0, 0].set_title("Top 20 Most Common Words", fontsize=16)
axs[0, 0].set_xlabel("Words", fontsize=14)
axs[0, 0].set_ylabel("Frequency", fontsize=14)
axs[0, 0].tick_params(axis='x', rotation=45, labelsize=10)

for i, count in enumerate(counts):
    axs[0, 0].text(i, count + 1, str(count), ha='center', fontsize=10)


st.write("#### Insights from Top 20 Most Common Words")
st.write("""
- The most frequent words in job descriptions often reflect key focus areas of the job market.
- For example, terms like 'Python', 'SQL', or 'analysis' may indicate high demand for data-related roles.
""")

# Salary Distribution by Experience Level Plot
sns.countplot(data=filtered_jobs, x='Experience_Level', hue='Salary_Range', ax=axs[0, 1], palette='coolwarm')
axs[0, 1].set_title("Salary Distribution by Experience Level", fontsize=16)
axs[0, 1].set_xlabel("Experience Level", fontsize=14)
axs[0, 1].set_ylabel("Count", fontsize=14)

for container in axs[0, 1].containers:
    axs[0, 1].bar_label(container, fontsize=10)


st.write("#### Insights from Salary Distribution")
st.write("""
- Entry-level positions often have a narrower salary range compared to senior roles, which have broader and higher salary brackets.
- Observe any unusual patterns, like overlapping ranges between different experience levels, which might indicate inconsistency in job categorization.
""")

# Most In-Demand Skills Plot
axs[1, 0].bar(skills, skill_counts, color='darkorange')
axs[1, 0].set_title("Most In-Demand Skills", fontsize=16)
axs[1, 0].set_xlabel("Skills", fontsize=14)
axs[1, 0].set_ylabel("Frequency", fontsize=14)
axs[1, 0].tick_params(axis='x', rotation=55, labelsize=10)

for i, count in enumerate(skill_counts):
    axs[1, 0].text(i, count + 1, str(count), ha='center', fontsize=10)


st.write("#### Insights from Most In-Demand Skills")
st.write("""
- The top 10 skills in demand show the technical competencies companies prioritize, such as programming languages and data tools.
- This helps candidates identify key areas to focus on for upskilling.
""")

# Sentiment Distribution Plot
sentiment_counts.plot(kind="bar", color=['lightgreen', 'lightcoral', 'lightgray'], ax=axs[1, 1])
axs[1, 1].set_title("Sentiment Distribution", fontsize=16)
axs[1, 1].set_xlabel("Sentiment", fontsize=14)
axs[1, 1].set_ylabel("Number of Job Descriptions", fontsize=14)

for i, value in enumerate(sentiment_counts.values):
    axs[1, 1].text(i, value + 1, str(value), ha='center', fontsize=10)


st.write("#### Insights from Sentiment Distribution")
st.write("""
- Sentiment analysis highlights the tone of job descriptions. Positive sentiments may indicate engaging and motivational descriptions, while negative sentiments might highlight concerns or strict requirements.
""")

plt.tight_layout()
st.pyplot(fig)

# Word Cloud Plot
st.subheader("Word Cloud for Job Descriptions")
wordcloud = WordCloud(width=1000, height=500, background_color='white').generate(" ".join(all_tokens))
fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
ax_wc.imshow(wordcloud, interpolation='bilinear')
ax_wc.axis("off")
ax_wc.set_title("Common Keywords in Job Descriptions", fontsize=16)
st.pyplot(fig_wc)


st.write("#### Insights from Word Cloud")
st.write("""
- The word cloud showcases keywords frequently used in job descriptions. Larger words represent higher occurrences.
- This provides a quick overview of what recruiters are emphasizing in their job postings.
""")

st.markdown("""
    <hr>
    <p style='text-align: center; font-size: 14px;'>Created by Team 1</p>
""", unsafe_allow_html=True)
