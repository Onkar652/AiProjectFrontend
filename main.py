import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from langchain_google_genai import GoogleGenerativeAI
from langchain.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

# --- Load secrets from Streamlit ---
db_url = st.secrets["DB_URL"]
google_api_key = st.secrets["GOOGLE_API_KEY"]

# Initialize SQLAlchemy engine and SQLDatabase
try:
    engine = create_engine(db_url)
    db = SQLDatabase(engine)
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

# Initialize Gemini model
try:
    llm = GoogleGenerativeAI(
        model="models/gemini-2.0-flash",
        google_api_key=google_api_key
    )
except Exception as e:
    st.error(f"LLM initialization failed: {e}")
    st.stop()

# Create SQL query chain
chain = create_sql_query_chain(llm, db)

# Token estimation
def estimate_tokens(text):
    return len(text) / 4

# Execute query
def execute_question(question):
    try:
        response = chain.invoke({"question": question})
        raw_output = response.strip()

        # Extract SQL if inside ```sql ... ```
        if "```sql" in raw_output:
            cleaned_sql = raw_output.split("```sql")[1].split("```")[0].strip()
        else:
            sql_lines = []
            for line in raw_output.splitlines():
                line = line.strip()
                if line.lower().startswith(("select", "with", "insert", "update", "delete")):
                    sql_lines.append(line)
            cleaned_sql = " ".join(sql_lines).strip()

        if not cleaned_sql:
            raise ValueError("No valid SQL found in LLM response.")

        with engine.connect() as conn:
            df = pd.read_sql(cleaned_sql, conn)

        return cleaned_sql, df, raw_output
    except Exception as e:
        st.error(f"Error executing query: {e}")
        return None, None, raw_output

# --- Streamlit UI ---
st.title("🧠 AI-Based SQL Query Builder")

# ---- Sidebar ----
st.sidebar.header("📚 Database Overview")
st.sidebar.markdown("""
Tables and sample columns:
- **employees**: id, name, age, department, salary  
- **departments**: id, name, location  
- **projects**: id, project_name, lead_id, budget  
- **sales**: id, employee_id, amount, date  
- **playing_with_neon**: id, name, value (sample random data)
""")

st.sidebar.header("💡 Suggested Queries")
suggested_queries = [
    "SELECT * FROM employees;",
    "SELECT name, salary FROM employees WHERE department='Engineering';",
    "SELECT * FROM projects;",
    "SELECT SUM(budget) FROM projects;",
    "SELECT employee_id, AVG(amount) as avg_sales FROM sales GROUP BY employee_id;",
    "SELECT * FROM playing_with_neon;"
]

# Make queries clickable
for query in suggested_queries:
    if st.sidebar.button(query):
        st.session_state.prefill_query = query

# Prefill input if clicked from sidebar
if "prefill_query" not in st.session_state:
    st.session_state.prefill_query = ""

question = st.text_input("Enter your question or requirement:", value=st.session_state.prefill_query)

# Clear history button
if st.button("Clear History"):
    st.session_state.history = []
    st.session_state.total_tokens = 0
    st.success("History cleared!")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

st.write(f"Total tokens used: {int(st.session_state.total_tokens)}")

# Execute query
if st.button("Execute") and question:
    cleaned_sql, result_df, raw_output = execute_question(question)
    if cleaned_sql and result_df is not None:
        prompt_tokens = estimate_tokens(question + cleaned_sql)
        st.session_state.total_tokens += prompt_tokens

        st.session_state.history.append({
            "question": question,
            "sql": cleaned_sql,
            "result": result_df,
            "tokens": prompt_tokens,
            "raw": raw_output
        })

# Display history and results
if st.session_state.history:
    st.subheader("📜 Query History")
    for idx, item in enumerate(reversed(st.session_state.history), start=1):
        with st.expander(f"{idx}. {item['question']}"):
            st.write("**Raw LLM Output:**")
            st.code(item["raw"])
            
            st.write("**SQL Query:**")
            st.code(item["sql"], language="sql")
            
            st.write("**Result:**")
            st.dataframe(item["result"], hide_index=True)

            # Show bar chart if numeric data exists
            if not item["result"].empty:
                st.write("**Bar Chart:**")
                try:
                    numeric_cols = item["result"].select_dtypes(include=["number"]).columns
                    if len(numeric_cols) > 0:
                        x_col = None
                        for col in item["result"].columns:
                            if col not in numeric_cols:
                                x_col = col
                                break
                        if x_col:
                            st.bar_chart(item["result"], x=x_col, y=numeric_cols[0])
                        else:
                            st.bar_chart(item["result"][numeric_cols])
                    else:
                        st.info("No numeric columns available for charting.")
                except Exception as e:
                    st.warning(f"Could not display chart: {e}")

            st.write("**Tokens Used:**", int(item["tokens"]))
            st.write("---")

    latest = st.session_state.history[-1]
    st.write("### Latest Query Result:")
    st.dataframe(latest["result"], hide_index=True)

    st.write("### Latest SQL Query:")
    st.code(latest["sql"], language="sql")

    if not latest["result"].empty:
        st.write("### Latest Query Bar Chart:")
        try:
            numeric_cols = latest["result"].select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 0:
                x_col = None
                for col in latest["result"].columns:
                    if col not in numeric_cols:
                        x_col = col
                        break
                if x_col:
                    st.bar_chart(latest["result"], x=x_col, y=numeric_cols[0])
                else:
                    st.bar_chart(latest["result"][numeric_cols])
            else:
                st.info("No numeric columns available for charting.")
        except Exception as e:
            st.warning(f"Could not display chart: {e}")

        # CSV Download
        csv = latest["result"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Result as CSV",
            data=csv,
            file_name="query_result.csv",
            mime="text/csv"
        )

        st.caption(f"Estimated Tokens Used: {int(latest['tokens'])}")

# Cost estimation
st.markdown("### 💰 Estimated Cost")
cost_per_k_token = 0.00025
total_cost = (st.session_state.total_tokens / 1000) * cost_per_k_token
st.write(f"**Total Cost:** ${total_cost:.4f}")
