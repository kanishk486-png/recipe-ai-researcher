import os
import tempfile
import streamlit as st
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool, AgentType

# Page Configuration
st.set_page_config(
    page_title="AI Recipe Research & Nutrition Engine",
    page_icon="🍳",
    layout="wide"
)

st.title("🍳 AI Culinary Research & Nutrition Engine")
st.markdown("Research recipes, cross-reference your personal cookbook files, and calculate exact nutrition profiles.")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("🔑 API Keys & Settings")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
spoonacular_key = st.sidebar.text_input("Spoonacular API Key (Optional)", type="password")

if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
if spoonacular_key:
    os.environ["SPOONACULAR_API_KEY"] = spoonacular_key

st.sidebar.divider()
st.sidebar.header("📚 Local Cookbook RAG")
uploaded_files = st.sidebar.file_uploader(
    "Upload your markdown/text recipe files", 
    type=["md", "txt"], 
    accept_multiple_files=True
)

# --- BACKEND FUNCTIONS ---

@st.cache_resource
-def get_vector_store_from_uploads(files_tuple):
    """Processes uploaded files into a temporary Chroma vector store."""
    if not files_tuple or not os.environ.get("OPENAI_API_KEY"):
        return None
    
    documents = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in files_tuple:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = TextLoader(temp_path)
            documents.extend(loader.load())
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(documents)
        
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(split_docs, embeddings)
        return vectorstore

def search_local_cookbook(query: str) -> str:
    """Searches user-uploaded cookbooks."""
    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
        return "No local cookbook files uploaded yet. Relying on web search."
    
    docs = st.session_state.vectorstore.similarity_search(query, k=3)
    return "\n---\n".join([d.page_content for d in docs])

def get_exact_nutrition(query: str) -> str:
    """Fetches real nutritional data via Spoonacular API."""
    api_key = os.environ.get("SPOONACULAR_API_KEY")
    if not api_key:
        return "Spoonacular API Key not provided. Skipping exact macro lookup."
    
    url = "https://api.spoonacular.com/recipes/parseIngredients"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    payload = {
        "ingredientList": query,
        "servings": 4,
        "includeNutrition": "true",
        "apiKey": api_key
    }
    
    try:
        response = requests.post(url, data=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            macros = []
            for item in data:
                name = item.get("name", "Ingredient")
                nutrition = item.get("nutrition", {})
                calories = next((n['amount'] for n in nutrition.get('nutrients', []) if n['name'] == 'Calories'), 0)
                macros.append(f"- {name}: {calories} kcal")
            return "Exact Nutritional API Breakdown:\n" + "\n".join(macros)
        return "Could not fetch exact macros from API."
    except Exception as e:
        return f"Nutrition API Error: {str(e)}"

# Initialize Vector Store if files are provided
if uploaded_files:
    # Use file names as a tuple cache key identifier
    file_names = tuple(f.name for f in uploaded_files)
    st.session_state.vectorstore = get_vector_store_from_uploads(file_names)
else:
    st.session_state.vectorstore = None

# --- MAIN INTERFACE ---
user_query = st.text_area(
    "What would you like to research?", 
    "Research authentic Neapolitan pizza dough, check my local cookbook for variations, and calculate nutrition metrics."
)

if st.button("Run Culinary Research", type="primary"):
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("⚠️ Please enter your OpenAI API Key in the sidebar to proceed.")
    else:
        with st.spinner("🤖 Chef Agent is actively researching history, local notes, and nutrition data..."):
            try:
                # Set up tools
                search_tool = DuckDuckGoSearchRun()
                tools = [
                    Tool(
                        name="LocalCookbookRAG",
                        func=search_local_cookbook,
                        description="Search uploaded personal markdown cookbooks for family recipes and custom notes."
                    ),
                    Tool(
                        name="ExactNutritionCalculator",
                        func=get_exact_nutrition,
                        description="Calculate exact calorie and macro counts using the Spoonacular nutrition API."
                    ),
                    Tool(
                        name="GeneralWebRecipeSearch",
                        func=search_tool.run,
                        description="Search the web for historical origins, cooking techniques, and culinary recipes."
                    )
                ]
                
                llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
                agent = initialize_agent(
                    tools=tools,
                    llm=llm,
                    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=True,
                    handle_parsing_errors=True
                )
                
                response = agent.run(user_query)
                
                st.success("Research Complete!")
                st.subheader("📋 Comprehensive Research Report")
                st.markdown(response)
                
            except Exception as e:
                st.error(f"An error occurred during execution: {e}")
