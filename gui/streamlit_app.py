import streamlit as st
import sys
import os
import uuid
import chromadb
from datetime import datetime
import json
from typing import List, Dict, Any
from collections import deque
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, SystemMessage
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from openai import OpenAIError
from langchain_core.runnables import RunnableLambda

load_dotenv()

# Movie context tracker class from chat_app.py
class MovieContext:
    def __init__(self, max_size=5):
        self._dq = deque(maxlen=max_size)
        self.movie_titles = [
            # From Marvel Fandom Hub:Movies
            "Howard the Duck","The Punisher","Captain America","The Fantastic Four","Blade","X-Men","Blade II","Spider-Man","Daredevil",
            "X2: X-Men United","Hulk","The Punisher","Spider-Man 2","Blade: Trinity","Elektra","Fantastic Four",
            "X-Men: The Last Stand","Ghost Rider","Spider-Man 3","Fantastic Four: Rise of the Silver Surfer",
            "Iron Man","The Incredible Hulk","Punisher: War Zone","X-Men Origins: Wolverine","Iron Man 2",
            "Thor","X-Men: First Class","Captain America: The First Avenger","Ghost Rider: Spirit of Vengeance",
            "The Avengers","The Amazing Spider-Man","Iron Man 3","The Wolverine","Thor: The Dark World",
            "Captain America: The Winter Soldier","The Amazing Spider-Man 2","X-Men: Days of Future Past",
            "Guardians of the Galaxy","Avengers: Age of Ultron","Ant-Man","Deadpool","X-Men: Apocalypse",
            "Captain America: Civil War","Doctor Strange","Logan","Guardians of the Galaxy Vol. 2",
            "Spider-Man: Homecoming","Thor: Ragnarok","Black Panther","Avengers: Infinity War","Deadpool 2","Ant-Man and the Wasp",
            "Captain Marvel","Avengers: Endgame","Dark Phoenix","Spider-Man: Far From Home","The New Mutants",
            "Black Widow","Shang-Chi and the Legend of the Ten Rings","Eternals","Spider-Man: No Way Home",
            "Doctor Strange in the Multiverse of Madness","Thor: Love and Thunder","Black Panther: Wakanda Forever",
            "Ant-Man and the Wasp: Quantumania","Guardians of the Galaxy Vol. 3","The Marvels","Deadpool & Wolverine",
            
            # From Wikipedia list
            "Man-Thing","Ghost Rider","Fantastic Four (2015)","Venom",
            "Morbius","Venom: Let There Be Carnage","Madame Web","Kraven the Hunter",
            
            # From MCU Fandom
            "Iron Man","The Incredible Hulk","Iron Man 2","Thor","Captain America: The First Avenger",
            "The Avengers","Iron Man 3","Thor: The Dark World","Captain America: The Winter Soldier",
            "Guardians of the Galaxy","Avengers: Age of Ultron","Ant-Man","Captain America: Civil War","Doctor Strange",  "Guardians of the Galaxy Vol. 2",
            "Spider-Man: Homecoming","Thor: Ragnarok","Black Panther","Avengers: Infinity War","Ant-Man and the Wasp","Captain Marvel","Avengers: Endgame","Spider-Man: Far From Home",
            "Black Widow","Shang-Chi and the Legend of the Ten Rings","Eternals","Spider-Man: No Way Home",
            "Doctor Strange in the Multiverse of Madness","Thor: Love and Thunder","Black Panther: Wakanda Forever",
            "Ant-Man and the Wasp: Quantumania","Guardians of the Galaxy Vol. 3","The Marvels","Deadpool & Wolverine"]

    def add(self, *movies):
        for m in movies:
            if m and m not in self._dq:
                self._dq.append(m)

    @property
    def list(self):
        return list(self._dq)

    def __bool__(self):
        return bool(self._dq)

    def extract_movies_from_query(self, query):
        """Extract movie titles from query using keyword matching"""
        query_lower = query.lower()
        found = []
        for title in self.movie_titles:
            if title in query_lower:
                found.append(title.title())
        return found

# Create global movie context instance
movie_context = MovieContext()

# Page config
st.set_page_config(
    page_title="Marvel Movies RAG Chat",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .session-list {
        max-height: 500px;
        overflow-y: auto;
        padding: 5px;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        background-color: #f8f9fa;
    }
    .stButton > button {
        width: 100%;
        margin-bottom: 5px;
        white-space: pre-line;
    }
    .main-header {
        text-align: center;
        color: #FF6B6B;
        margin-bottom: 20px;
    }
    /* Remove red background from multiselect selected items */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: transparent !important;
        border-color: black !important;
    }
    .stMultiSelect [data-baseweb="tag"] span {
        color: black !important;
    }
    /* Style multiselect dropdown */
    .stMultiSelect [data-baseweb="select"] {
        border-color: #e0e0e0 !important;
    }
    /* Move multiselect label higher */
    .stMultiSelect > label {
        margin-bottom: 8px !important;
        margin-top: -5px !important;
    }
    /* Make chat input sticky at bottom with enhanced styling */
    .stChatInput {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        z-index: 1000 !important;
        padding: 15px 20px 20px 20px !important;
    }
    /* Ensure chat input container takes most of main content area width */
    .stChatInput > div {
        max-width: none !important;
        width: 100% !important;
        margin: 0 auto !important;
        max-width: calc(54% - 40px) !important;
        margin-left: calc(23% + 20px) !important;
        margin-right: calc(23% + 20px) !important;
    }
    /* Style the actual input field to fit properly */
    .stChatInput input[type="text"] {
        border-radius: 30px !important;
        padding: 15px 25px !important;
        font-size: 16px !important;
        box-shadow: 0 2px 10px rgba(255, 107, 107, 0.2) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }
    .stChatInput input[type="text"]:focus {
        border-color: #FF4444 !important;
        box-shadow: 0 0 0 4px rgba(255, 107, 107, 0.3) !important;
        outline: none !important;
    }
    /* Add padding to main content to prevent overlap */
    .main .block-container {
        padding-bottom: 140px !important;
        max-height: calc(100vh - 140px) !important;
        overflow-y: auto !important;
    }
    /* Make left sidebar scrollable */
    .element-container:has([data-testid="column"]:first-child) {
        max-height: calc(100vh - 140px) !important;
        overflow-y: auto;
        padding-right: 10px;
    }
    /* Custom scrollbar for left sidebar */
    .element-container:has([data-testid="column"]:first-child)::-webkit-scrollbar {
        width: 8px;
    }
    .element-container:has([data-testid="column"]:first-child)::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    .element-container:has([data-testid="column"]:first-child)::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    .element-container:has([data-testid="column"]:first-child)::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    /* Hide streamlit footer */
    .stApp > footer {
        display: none !important;
    }
    /* Ensure proper spacing for chat messages */
    .stChatMessage {
        margin-bottom: 15px !important;
    }
    /* Make sure content doesn't get hidden behind sticky input */
    .element-container:last-child {
        margin-bottom: 20px !important;
    }
    /* Make all logging messages smaller with small padding and margin */
    .stAlert, .stSuccess, .stError, .stInfo, .stWarning {
        font-size: 11px !important;
        padding: 4px 8px !important;
        margin: 2px 0 !important;
        line-height: 1.2 !important;
    }
    .stAlert > div, .stSuccess > div, .stError > div, .stInfo > div, .stWarning > div {
        padding: 4px 8px !important;
        margin: 2px 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Setup paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import after path setup
from scripts.process_data import LocalServerEmbeddings, EMBED_ENDPOINT

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = f"session_{uuid.uuid4().hex[:8]}"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = None
if 'rag_chain_local' not in st.session_state:
    st.session_state.rag_chain_local = None
if 'rag_chain_openai' not in st.session_state:
    st.session_state.rag_chain_openai = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'current_datasets' not in st.session_state:
    st.session_state.current_datasets = None
if 'last_retrieval_scores' not in st.session_state:
    st.session_state.last_retrieval_scores = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Phi-4 Mini"
if 'left_sidebar_open' not in st.session_state:
    st.session_state.left_sidebar_open = True
if 'right_sidebar_open' not in st.session_state:
    st.session_state.right_sidebar_open = True

# Firestore setup
PROJECT_ID = "fml-project-5509d"
FIRESTORE_COLLECTION = "chat_history"

@st.cache_resource
def get_firestore_client():
    return firestore.Client(project=PROJECT_ID)

def get_all_sessions():
    """Get all chat sessions from Firestore"""
    try:
        client = get_firestore_client()
        sessions = client.collection(FIRESTORE_COLLECTION).list_documents()
        session_data = []
        
        for session in sessions:
            session_doc = session.get()
            if session_doc.exists:
                data = session_doc.to_dict()
                session_data.append({
                    'session_id': session.id,
                    'created_at': data.get('created_at', datetime.now()),
                    'message_count': len(data.get('messages', [])),
                    'last_message': data.get('messages', [])[-1] if data.get('messages') else None
                })
        
        # Sort by created_at in ascending order (oldest first, which will be last in display)
        return sorted(session_data, key=lambda x: x['created_at'], reverse=False)
    except Exception as e:
        st.error(f"Error loading sessions: {e}")
        return []

def load_session_messages(session_id: str):
    """Load messages from a specific session"""
    try:
        client = get_firestore_client()
        chat_history = FirestoreChatMessageHistory(
            session_id=session_id,
            collection=FIRESTORE_COLLECTION,
            client=client,
        )
        
        # Get messages from Firestore
        messages = chat_history.messages
        
        # Convert to Streamlit format
        streamlit_messages = []
        for i, message in enumerate(messages):
            if hasattr(message, 'content'):
                # Determine role based on message type
                if hasattr(message, 'type'):
                    role = "user" if message.type == "human" else "assistant"
                else:
                    # Fallback: odd indices are human, even are AI
                    role = "user" if i % 2 == 0 else "assistant"
                
                streamlit_messages.append({
                    "role": role,
                    "content": message.content
                })
        
        return streamlit_messages, chat_history
    except Exception as e:
        st.error(f"Error loading session {session_id}: {e}")
        return [], None

def setup_embedder_and_retriever(datasets: List[str]):
    """Setup embedder and retriever for selected datasets"""
    embedder = LocalServerEmbeddings(
        endpoint=EMBED_ENDPOINT,
        model="gaianet/Nomic-embed-text-v1.5-Embedding-GGUF"
    )
    
    all_retrievers = []
    total_docs = 0
    
    for dataset in datasets:
        chroma_dir = os.path.join(project_root, "data", "chroma", dataset)
        try:
            client = chromadb.PersistentClient(path=chroma_dir)
            collection = client.get_collection(name="marvel_films")
            doc_count = collection.count()
            total_docs += doc_count
            
            vectordb = Chroma(
                client=client,
                collection_name="marvel_films",
                embedding_function=embedder,
            )
            
            retriever = vectordb.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10}
            )
            all_retrievers.append((retriever, dataset, doc_count))
            
        except Exception as e:
            st.error(f"Error setting up {dataset}: {e}")
    
    return all_retrievers, total_docs, embedder

def calculate_simple_boost(doc_metadata: dict, context_movies: List[str]) -> float:
    """Simplified boost calculation - returns additive value instead of multiplier"""
    if not doc_metadata or not context_movies:
        return 0.0
    
    doc_title = doc_metadata.get('title', '').lower()
    
    for movie in context_movies:
        if movie.lower() in doc_title or doc_title in movie.lower():
            return 0.3
    
    return 0.0

def create_combined_retriever(retrievers_list):
    """Create a combined retriever that sorts all documents globally and keeps top 8"""
    def combined_retrieve(query):
        # Handle both string queries and dictionary inputs from the RAG chain
        if isinstance(query, dict):
            actual_query = query.get('input', '')
        elif isinstance(query, str):
            actual_query = query
        else:
            actual_query = str(query)
        
        if not actual_query:
            return []
        
        # Extract movies from query and update context
        detected_movies = movie_context.extract_movies_from_query(actual_query)
        if detected_movies:
            movie_context.add(*detected_movies)
        
        # Get context movies for boosting
        context_movies = movie_context.list
        
        all_docs_with_scores = []
        
        # Collect documents from all retrievers
        for retriever, dataset_name, _ in retrievers_list:
            try:
                vectordb = retriever.vectorstore
                docs_with_scores = vectordb.similarity_search_with_relevance_scores(actual_query, k=15)
                
                for doc, score in docs_with_scores:
                    if score > 0.3:  # Basic threshold filter
                        # Add movie boost instead of multiply
                        movie_boost_value = calculate_simple_boost(doc.metadata, context_movies)
                        boosted_score = score + movie_boost_value
                        
                        # Add metadata
                        if 'dataset' not in doc.metadata:
                            doc.metadata['dataset'] = dataset_name
                        doc.metadata['original_score'] = score
                        doc.metadata['movie_boost'] = movie_boost_value
                        doc.metadata['final_score'] = boosted_score
                        
                        all_docs_with_scores.append((doc, boosted_score))
                        
            except Exception as e:
                st.warning(f"Error retrieving from {dataset_name}: {e}")
        
        # Sort all documents globally by final score (highest first)
        all_docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top 3 documents (reduced from 20 to fit token limit)
        top_docs = [doc for doc, score in all_docs_with_scores[:3]]
        
        return top_docs
    
    return RunnableLambda(combined_retrieve)

def setup_rag_chains(retriever):
    """Setup RAG chain with history-aware retriever and proper prompts"""
    
    # Create a history-aware retriever that can reformulate queries based on chat history
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference "
        "the chat history, formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just reformulate it "
        "if needed and otherwise return it as is. If the question contains pronouns "
        "(he, she, they, etc.), replace them with the appropriate character names "
        "from the chat history."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    qa_system_prompt = (
        """
            <role>
                You are a Marvel movies question-answering assistant. Your sole purpose is to answer questions about Marvel films using ONLY the provided context documents and conversation history.
            </role>

            <critical_rules>
                ABSOLUTE RULE: If the specific information is not explicitly stated in the context documents below, you MUST respond with "I don't have that information in the provided context."
                
                FORBIDDEN: You are STRICTLY PROHIBITED from using any knowledge from your training data about Marvel movies that is not present in the context.
                
                FORBIDDEN: Do not make up, infer, or hallucinate ANY details about plot points, character relationships, or events that are not explicitly mentioned in the context.
            </critical_rules>

            <instructions>
                <primary_rule>
                ONLY use information from the retrieved context below and the conversation history. NEVER use your own knowledge about Marvel movies, characters, or plots.
                </primary_rule>

                <response_guidelines>
                - If the answer is clearly present in the context: Provide a clear, direct answer in 1-3 sentences
                - Synthesize information from multiple context documents when they all relate to the same question
                - Use conversation history to resolve pronouns (he, she, his, her, they, etc.) - look at recent messages to understand who is being referenced
                - If multiple movies are mentioned in the context, clearly specify which information comes from which movie
                - Be direct and concise when the information is clearly present
                - Quote specific details from the context when relevant
                </response_guidelines>

                <synthesis_instructions>
                - Look across ALL context documents for relevant information
                - If multiple documents mention the same character/topic, combine that information
                - If the question asks "what movies does X appear in?" and multiple documents reference different movies with that character, list all the movies mentioned
                - Don't be overly cautious - if the information is there, use it confidently
                </synthesis_instructions>

                <pronoun_resolution>
                - Use recent conversation to understand pronoun references
                - If someone asks "who is his daughter?" after discussing Tony Stark, "his" refers to Tony Stark
                - If someone asks "how does he lose?" after discussing Thanos, "he" refers to Thanos
                - BUT only answer using information from the retrieved context documents
                </pronoun_resolution>

                <forbidden_actions>
                - Do not add information from your training data that isn't in the context
                - Do not make educated guesses beyond reasonable pronoun resolution and synthesis
                - Do not provide partial answers mixed with your own knowledge
                - Do not suggest where to find additional information
                </forbidden_actions>
            </instructions>

            <context>
            {context}
            </context>

            <output_format>
            Answer the question using the context above and conversation history for pronoun resolution. Synthesize information from multiple documents when relevant. Be direct and concise.
            </output_format>        
        """
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    return contextualize_q_prompt, qa_prompt

def get_llm(model_choice: str):
    """Get LLM based on model choice"""
    if model_choice == "Gemma 4B":
        return ChatOpenAI(
        model_name       = os.environ["HF_ENDPOINT_MODEL"],     
        openai_api_base  = os.environ["HF_ENDPOINT_URL"],
        openai_api_key   = os.environ["HF_API_TOKEN"],
        temperature      = 0.0,
        max_tokens       = 128,
    )
    else:
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.0,
        )

# Main App
def main():

    # Create three columns: left sidebar, main content, right sidebar
    left_sidebar, main_content, right_sidebar = st.columns([1, 3, 1])
    
    # Left Sidebar - Sessions
    with left_sidebar:
        st.markdown("# Sessions")
        
        # New session button
        if st.button("New Chat", type="primary", use_container_width=True):
            st.session_state.current_session_id = f"session_{uuid.uuid4().hex[:8]}"
            st.session_state.messages = []
            st.session_state.chat_history = None
            st.rerun()
        
        st.markdown("### All Sessions")
        
        # Load existing sessions
        sessions = get_all_sessions()
        if sessions:
            # Create a container for scrollable sessions
            with st.container():
                for i, session in enumerate(sessions):
                    session_id = session['session_id']
                    message_count = session['message_count']

                    # Highlight current session
                    if session_id == st.session_state.current_session_id:
                        button_type = "primary"
                        label = f"{session_id[:8]}... ({message_count})"
                    else:
                        button_type = "secondary"
                        label = f"{session_id[:8]}... ({message_count})"
                    
                    if st.button(label, key=f"session_{i}", type=button_type, use_container_width=True):
                        if session_id != st.session_state.current_session_id:
                            with st.spinner(f"Loading session..."):
                                # Load messages from the selected session
                                loaded_messages, loaded_chat_history = load_session_messages(session_id)
                                
                                # Update session state
                                st.session_state.current_session_id = session_id
                                st.session_state.messages = loaded_messages
                                st.session_state.chat_history = loaded_chat_history
                                
                                st.success(f"Loaded {len(loaded_messages)} messages")
                                st.rerun()
            
        else:
            st.info("No previous sessions found")
        
    
    # Right Sidebar - Database Configuration
    with right_sidebar:
        st.markdown("# Databases")
        
        # Dataset Selection
        datasets = st.multiselect(
            "Select Datasets:",
            ["chroma_wikipedia", "chroma_marvel_wiki", "chroma_mcu_wiki"],
            default=["chroma_wikipedia"],
            help="Choose knowledge bases to search"
        )
        
        if not datasets:
            st.error("Select at least one dataset!")
            return
        
        # Check if datasets changed
        if st.session_state.current_datasets != datasets:
            st.session_state.current_datasets = datasets
            st.session_state.rag_chain_local = None
            st.session_state.rag_chain_openai = None
            st.session_state.retriever = None
        
        # Dataset Info
        st.markdown("### Dataset Info")
        for dataset in datasets:
            chroma_dir = os.path.join(project_root, "data", "chroma", dataset)
            try:
                client = chromadb.PersistentClient(path=chroma_dir)
                collection = client.get_collection(name="marvel_films")
                doc_count = collection.count()
                st.success(f"**{dataset}**\n{doc_count:,} documents")
            except Exception as e:
                st.error(f"**{dataset}**\nConnection error")
        
    
    # Main Content Area
    with main_content:
        # Header
        st.markdown('<h1 style="text-align: center; color: #FF6B6B;">Marvel Movies RAG Chat</h1>', unsafe_allow_html=True)
        # Model Selection in main area
        col_model1, col_model2 = st.columns(2)
        
        # Initialize selected model if not set
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = "Gemma 4B"
        
        with col_model1:
            button_type = "primary" if st.session_state.selected_model == "Gemma 4B" else "secondary"
            if st.button("Gemma 4B", type=button_type, use_container_width=True):
                st.session_state.selected_model = "Gemma 4B"
                st.rerun()
        
        with col_model2:
            button_type = "primary" if st.session_state.selected_model == "GPT-3.5 turbo" else "secondary"
            if st.button("GPT-3.5 Turbo", type=button_type, use_container_width=True):
                st.session_state.selected_model = "GPT-3.5 turbo"
                st.rerun()
        
        model_choice = st.session_state.selected_model
        
        st.markdown("---")
        
        # Setup components if needed
        if (st.session_state.rag_chain_local is None or 
            st.session_state.rag_chain_openai is None or 
            st.session_state.retriever is None):
            
            with st.spinner("Setting up AI components..."):
                try:
                    # Setup retrievers
                    retrievers_list, total_docs, embedder = setup_embedder_and_retriever(datasets)
                    if retrievers_list:
                        st.session_state.retriever = create_combined_retriever(retrievers_list)
                        st.success(f"Connected to {len(datasets)} dataset(s) with {total_docs} total documents")
                    
                    # Setup RAG components
                    contextualize_q_prompt, qa_prompt = setup_rag_chains(st.session_state.retriever)
                    st.session_state.contextualize_q_prompt = contextualize_q_prompt
                    st.session_state.qa_prompt = qa_prompt
                    st.session_state.rag_chain_local = True
                    st.session_state.rag_chain_openai = True
                    st.success("RAG components initialized")
                    
                    # Setup chat history
                    if st.session_state.chat_history is None:
                        client = get_firestore_client()
                        st.session_state.chat_history = FirestoreChatMessageHistory(
                            session_id=st.session_state.current_session_id,
                            collection=FIRESTORE_COLLECTION,
                            client=client,
                        )
                    
                except Exception as e:
                    st.error(f"Error setting up components: {e}")
                    return
        
        # Chat interface        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message:
                    with st.expander("Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**Source {i}:** {source}")
        
        # Chat input - always at the bottom
        if prompt := st.chat_input("Ask me anything about Marvel movies!"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Add to Firestore history
            st.session_state.chat_history.add_user_message(prompt)
            
            # Show retrieved documents
            with st.expander("Retrieved Documents"):
                retrieved_docs = st.session_state.retriever.invoke(prompt)
                
                # Show movie context information
                if movie_context:
                    st.info(f"**Movie Context:** {', '.join(movie_context.list)}")
                
                # Show detected movies from current query
                detected_movies = movie_context.extract_movies_from_query(prompt)
                if detected_movies:
                    st.success(f"**Detected Movies in Query:** {', '.join(detected_movies)}")
                
                # Show dataset summary
                dataset_counts = {}
                for doc in retrieved_docs:
                    dataset = doc.metadata.get('dataset', 'Unknown')
                    dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
                
                st.write("**Search Summary:**")
                for dataset, count in dataset_counts.items():
                    st.write(f"  • {dataset}: {count} documents")
                st.write("---")
                
                st.write(f"**Retrieved {len(retrieved_docs)} documents:**")
                
                # Display documents with enhanced metadata
                for i, doc in enumerate(retrieved_docs[:10]):
                    st.write(f"**#{i+1} - {doc.metadata.get('dataset', 'Unknown')}**")
                    st.write(f"{doc.page_content}")
                    if 'title' in doc.metadata:
                        st.write(f"*Title: {doc.metadata['title']}*")
                    # Show scoring information if available
                    if 'original_score' in doc.metadata:
                        original = doc.metadata.get('original_score', 0)
                        boost = doc.metadata.get('movie_boost', 0)
                        final = doc.metadata.get('final_score', 0)
                        st.write(f"*Scores: Original={original:.3f}, Boost={boost:.3f}, Final={final:.3f}*")
                    st.write("---")
            
            # Generate responses based on model choice
            if model_choice == "Gemma 4B":
                with st.chat_message("assistant"):
                    with st.spinner("Local model thinking..."):
                        try:
                            # Get the selected LLM
                            llm = get_llm(model_choice)
                            
                            # Create history-aware retriever
                            history_aware_retriever = create_history_aware_retriever(
                                llm, st.session_state.retriever, st.session_state.contextualize_q_prompt
                            )
                            
                            # Create RAG chain
                            question_answer_chain = create_stuff_documents_chain(llm, st.session_state.qa_prompt)
                            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
                            
                            # Get response
                            result = rag_chain.invoke({
                                "input": prompt, 
                                "chat_history": st.session_state.chat_history.messages
                            })
                            
                            st.markdown(f"**Gemma 4B Response:**\n\n{result['answer']}")
                            
                            # Add to session state and Firestore
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": f"**Gemma 4B Response:**\n\n{result['answer']}"
                            })
                            st.session_state.chat_history.add_ai_message(result['answer'])
                            
                        except Exception as e:
                            error_msg = f"Local model error: {e}"
                            st.error(error_msg)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": error_msg
                            })
            
            elif model_choice == "GPT-3.5 turbo":
                with st.chat_message("assistant"):
                    with st.spinner("OpenAI thinking..."):
                        try:
                            # Get the selected LLM
                            llm = get_llm(model_choice)
                            
                            # Create history-aware retriever
                            history_aware_retriever = create_history_aware_retriever(
                                llm, st.session_state.retriever, st.session_state.contextualize_q_prompt
                            )
                            
                            # Create RAG chain
                            question_answer_chain = create_stuff_documents_chain(llm, st.session_state.qa_prompt)
                            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
                            
                            # Get response
                            result = rag_chain.invoke({
                                "input": prompt, 
                                "chat_history": st.session_state.chat_history.messages
                            })
                            
                            st.markdown(f"**GPT-3.5 Turbo Response:**\n\n{result['answer']}")
                            
                            # Add to session state and Firestore
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": f"**GPT-3.5 Turbo Response:**\n\n{result['answer']}"
                            })
                            st.session_state.chat_history.add_ai_message(result['answer'])
                            
                        except OpenAIError as e:
                            error_msg = f"OpenAI error: {e}"
                            st.error(error_msg)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": error_msg
                            })

if __name__ == "__main__":
    main() 