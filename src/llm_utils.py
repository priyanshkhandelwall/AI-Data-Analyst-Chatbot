import os
import json
from groq import Groq
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent

# ---------------- LLM CLIENT ----------------
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def safe_json(data):
    """Safely serialize pandas / numpy objects."""
    return json.dumps(data, default=str)


# ---------------- BASIC LLM CALL ----------------
def invoke_llm(system_prompt, user_message, history=None, response_format=None):
    try:
        messages = [{"role": "system", "content": system_prompt}]

        if history:
            for msg in history:
                messages.append(
                    {"role": msg["role"], "content": msg["content"]}
                )

        messages.append({"role": "user", "content": user_message})

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"LLM error: {e}")
        return None


# ---------------- LANGCHAIN PANDAS AGENT ----------------
def get_pandas_agent(df):
    llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)


    return create_pandas_dataframe_agent(
    llm,
    df,
    verbose=False,
    allow_dangerous_code=True,
)



# ---------------- QUERY ANALYSIS ----------------
def analyze_query(user_query, df_schema, history=None):
    system_prompt = f"""
You are a data analyst assistant.
Identify metrics, filters, and intent from the user's query.

Dataset schema:
{safe_json(df_schema)}

Respond in JSON with keys:
metrics (array), filters (array), intent (string).
"""
    result = invoke_llm(system_prompt, user_query, history)

    try:
        return json.loads(result)
    except Exception:
        return {
            "metrics": [],
            "filters": [],
            "intent": "unknown",
        }

# ---------------- RESULT EXPLANATION ----------------
def explain_results(user_query, analysis_data, df_schema, history=None):
    system_prompt = f"""
You are a data analyst assistant.
Explain the analysis results clearly in plain language.

Dataset schema:
{safe_json(df_schema)}
"""

    user_message = f"""
User query:
{user_query}

Analysis results:
{safe_json(analysis_data)}
"""

    result = invoke_llm(system_prompt, user_message, history)

    return {
        "summary": result or "Analysis complete.",
        "keyFindings": [],
    }


# ---------------- STATISTICAL TEST SUGGESTION ----------------
def suggest_statistical_test(user_query, column_stats, df_schema, history=None):
    system_prompt = f"""
You are a statistical consultant.
Recommend an appropriate statistical test.

Dataset schema:
{safe_json(df_schema)}

Respond in JSON with:
testName, explanation, requirements, interpretation.
"""

    user_message = f"""
User query:
{user_query}

Column statistics:
{safe_json(column_stats)}
"""

    result = invoke_llm(system_prompt, user_message, history)

    try:
        return json.loads(result)
    except Exception:
        return {
            "testName": "Descriptive Statistics",
            "explanation": "Start with descriptive statistics.",
            "requirements": [],
            "interpretation": "Review distributions and summary metrics.",
        }
