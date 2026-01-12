import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
import time

from analysis import (
    get_data_profile,
    get_column_stats,
    get_correlations,
    detect_outliers,
)
from llm_utils import (
    analyze_query,
    explain_results,
    suggest_statistical_test,
    get_pandas_agent,
)

st.set_page_config(page_title="Data Analyst Chatbot", layout="wide")


def main():
    st.title("📊 Data Analyst Chatbot")
    st.markdown("Upload your data and ask questions to get insights.")

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("Data Source")
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

        show_code = st.checkbox("Show Agent Code", value=False)

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        if uploaded_file is None:
            st.info("Using sample data. Upload your own to analyze.")
            sample_path = os.path.join(
                os.path.dirname(__file__), "..", "data", "sample_data.csv"
            )
            if os.path.exists(sample_path):
                df = pd.read_csv(sample_path)
            else:
                df = pd.DataFrame(
                    {
                        "Date": pd.date_range(start="2023-01-01", periods=10),
                        "Sales": [100, 120, 150, 130, 160, 180, 200, 190, 210, 230],
                        "Category": ["A", "B"] * 5,
                    }
                )
        else:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success(f"Loaded: {uploaded_file.name}")

    # ---------------- Tabs ----------------
    tab1, tab2, tab3 = st.tabs(
        ["💬 Chat & Analysis", "📋 Data Profile", "📈 Visualizations"]
    )

    # ---------------- Chat Tab ----------------
    with tab1:
        st.header("Chat with your Data")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "agent" not in st.session_state:
            try:
                st.session_state.agent = get_pandas_agent(df)
            except Exception:
                st.session_state.agent = None

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about your data..."):
            st.session_state.messages.append(
                {"role": "user", "content": prompt}
            )

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    agent = get_pandas_agent(df)
                    try:
                        if st.session_state.agent is None:
                            raise RuntimeError("LLM agent unavailable")

                        if "distribution" in prompt.lower() or "plot" in prompt.lower():
                            st.info("Use the Visualizations tab for charts.")
                            full_response = "I’ve prepared the chart in the Visualizations tab."
                        else:
                             agent_response = agent.invoke({"input": prompt})
                             full_response = agent_response["output"]


                        def stream_text(text):
                            for word in text.split():
                                yield word + " "
                                time.sleep(0.02)

                        st.write_stream(stream_text(full_response))

                        if show_code:
                            st.info(
                                "Code execution is handled internally by the Pandas Agent."
                            )

                        st.session_state.messages.append(
                            {"role": "assistant", "content": full_response}
                        )

                    except Exception as e:
                        st.warning(
                            "LLM unavailable. Falling back to basic analysis."
                        )

                        profile = get_data_profile(df)
                        schema = profile["columns"]
                        stats_data = get_column_stats(df)

                        query_analysis = analyze_query(
                            prompt,
                            schema,
                            history=st.session_state.messages[:-1],
                        )

                        analysis_payload = {
                            "queryAnalysis": query_analysis,
                            "columnStats": stats_data,
                        }

                        explanation = explain_results(
                            prompt,
                            analysis_payload,
                            schema,
                            history=st.session_state.messages[:-1],
                        )

                        fallback_text = explanation.get(
                            "summary", "Analysis complete."
                        )

                        st.markdown(fallback_text)

                        st.session_state.messages.append(
                            {"role": "assistant", "content": fallback_text}
                        )

            if st.button("Suggest Statistical Test"):
                stats_data = get_column_stats(df)
                profile = get_data_profile(df)

                test_rec = suggest_statistical_test(
                    prompt,
                    stats_data,
                    profile["columns"],
                    history=st.session_state.messages[:-1],
                )

                st.info(
                    f"**Recommended Test:** {test_rec['testName']}\n\n{test_rec['explanation']}"
                )

    # ---------------- Data Profile Tab ----------------
    with tab2:
        st.header("Data Profile & Quality")

        profile = get_data_profile(df)

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", profile["rowCount"])
        c2.metric("Columns", profile["columnCount"])
        c3.metric("Duplicates", profile["duplicateRows"])

        st.subheader("Column Information")
        st.table(pd.DataFrame(profile["columns"]))

        st.subheader("Missing Values")
        st.bar_chart(pd.Series(profile["missingValues"]))

        st.subheader("Outlier Detection")
        outliers = detect_outliers(df)
        if outliers:
            st.dataframe(pd.DataFrame(outliers))
        else:
            st.write("No numeric columns for outlier detection.")


    # ---------------- Visualization Tab ----------------
    with tab3:
        st.header("Interactive Visualizations")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        if numeric_cols:
            col = st.selectbox(
                "Select column for distribution", numeric_cols
            )
            fig = px.histogram(
                df, x=col, marginal="box", title=f"Distribution of {col}"
            )
            st.plotly_chart(fig, use_container_width=True)

            if len(numeric_cols) >= 2:
                corr = df[numeric_cols].corr()
                fig = px.imshow(
                    corr,
                    text_auto=True,
                    title="Correlation Heatmap",
                    color_continuous_scale="RdBu_r",
                )
                st.plotly_chart(fig, use_container_width=True)

        if categorical_cols and numeric_cols:
            cat = st.selectbox("Category", categorical_cols)
            num = st.selectbox("Value", numeric_cols)
            fig = px.box(
                df,
                x=cat,
                y=num,
                color=cat,
                title=f"{num} by {cat}",
            )
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
