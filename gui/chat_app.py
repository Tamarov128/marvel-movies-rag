import sys, os
import uuid
from typing import List
from collections import deque
import chromadb
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
import json
from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from openai import OpenAI as OpenAIClient
import time

load_dotenv()
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Set up firebase
PROJECT_ID = "fml-project-5509d"
# Generate a new session ID each time to avoid corrupted history
SESSION_ID = f"user_session_{int(time.time())}_{uuid.uuid4().hex[:8]}"
COLLECTION_NAME = "chat_history"

print("Initializing Firestore Client...")
client = firestore.Client(project=PROJECT_ID)

print(f"Initializing Firestore Chat Message History with session ID: {SESSION_ID}")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)

from scripts.process_data import LocalServerEmbeddings, EMBED_ENDPOINT

# Movie context tracker class
class MovieContext:
    def __init__(self, max_size=5):
        self._dq = deque(maxlen=max_size)
        self.movie_titles = marvel_movies = [
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

def setup_embedder_and_retriever(datasets: List[str]):
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
            print(f"Error setting up {dataset}: {e}")

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
                print(f"Error retrieving from {dataset_name}: {e}")
        
        # Sort all documents globally by final score (highest first)
        all_docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top 3 documents (reduced from 20 to fit token limit)
        top_docs = [doc for doc, score in all_docs_with_scores[:3]]
        
        return top_docs
    
    return RunnableLambda(combined_retrieve)

def setup_rag_chains(retriever):
    llm = get_llm()

    # Create a history-aware retriever that can reformulate queries based on chat history
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference "
        "the chat history, formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just reformulate it "
        "if needed and otherwise return it as is. If the question contains pronouns "
        "(he, she, they, etc.), replace them with the appropriate character names "
        "from the chat history."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

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

    # Create a prompt template for answering questions
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create the question-answering chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create the complete RAG chain with history-aware retriever
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

def get_llm():
    """Get LLM based on model choice"""

    return ChatOpenAI(
        model_name       = os.environ["HF_ENDPOINT_MODEL"],     
        openai_api_base  = os.environ["HF_ENDPOINT_URL"],
        openai_api_key   = os.environ["HF_API_TOKEN"],
        temperature      = 0.0,
        max_tokens       = 128,
    )
    # Alternative options (comment out the above and uncomment one below)
    # 
    # # Use OpenAI GPT
    # return ChatOpenAI(
    #     model="gpt-3.5-turbo",
    #     temperature=0.0,
    # )
    # 
    # # Use local LM Studio
    # return ChatOpenAI(
    #     model_name="llama-3.1-tulu-3-8b",
    #     openai_api_base = "http://127.0.0.1:1234/v1",
    #     openai_api_key = "lm-studio",    
    # )

def setup_rag_pipeline():
    """Set up the complete RAG pipeline"""
    print("Setting up RAG pipeline...")
    
    try:
        datasets = ["chroma_wikipedia", "chroma_marvel_wiki", "chroma_mcu_wiki"]
        retrievers_list, total_docs, embedder = setup_embedder_and_retriever(datasets)
        
        if not retrievers_list:
            print("No datasets available. Please check your data directory.")
            return None, None, 0
            
        retriever = create_combined_retriever(retrievers_list)
        rag_chain = setup_rag_chains(retriever)
        
        print(f"Connected to {len(datasets)} dataset(s) with {total_docs} total documents")
        return retriever, rag_chain, total_docs
        
    except Exception as e:
        print(f"Error setting up RAG pipeline: {e}")
        return None, None, 0

def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    print("=" * 50)
    
    # Use shared pipeline setup
    retriever, rag_chain, total_docs = setup_rag_pipeline()
    if rag_chain is None:
        return
    
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        
        # Add user message to chat history
        chat_history.add_user_message(query)
        
        if movie_context:
            print(f"Movie context: {', '.join(movie_context.list)}")
        
        print("Generating response...")
        
        try:
            # Process the user's query through the retrieval chain
            result = rag_chain.invoke({"input": query, "chat_history": chat_history.messages})
            
            # Display retrieved documents for debugging
            if 'context' in result and result['context']:
                print(f"\nRetrieved {len(result['context'])} documents:")
                for i, doc in enumerate(result['context'][:5], 1):  # Show first 5 docs
                    title = doc.metadata.get('title', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
                    content = doc.page_content[:150] if hasattr(doc, 'page_content') else str(doc)[:150]
                    print(f"  {i}. {title}: {content}...")
                print()
            
            # Display the AI's response
            print(f"AI: {result['answer']}")
            
            # Add AI response to chat history
            chat_history.add_ai_message(result["answer"])
            
        except Exception as e:
            print(f"Error generating response: {e}")
            print("Please try again with a different question.")
        
        print("=" * 50)

def main():
    """Main function to run the chat interface"""
    print("Starting Marvel Movies RAG Chat Interface...")
    
    print("\nChoose an option:")
    print("1. Start chat interface")
    print("2. Test system with GPT-4 evaluation")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1, 2, or 3): ").strip()
        
        if choice == "1":
            print("Starting chat interface...")
            continual_chat()
            break
        elif choice == "2":
            print("Running evaluation tests...")
            test_with_evaluation()
            
            # Ask if they want to continue with chat
            continue_choice = input("\nDo you want to start the chat interface now? (y/n): ")
            if continue_choice.lower() in ['y', 'yes']:
                print("Starting chat interface...")
                continual_chat()
            break
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

# Judge prompt template for evaluation
JUDGE_PROMPT = """
You are an impartial evaluator of Marvel movie question-answering systems. Analyze the following interaction:

**User Query**: {query}

**Retrieved Context**:
{context}

**System Response**: {response}

**Chat History Context**: {chat_history}

Evaluate based on these criteria (score 1-5 for each):
1. **Retrieval Accuracy**: Were the most relevant documents retrieved? (5 = perfect match, 1 = completely irrelevant)
2. **Answer Relevance**: Did the answer directly address the query? (5 = directly answers, 1 = doesn't address query)
3. **Context Utilization**: Was the answer properly grounded in the provided context? (5 = fully grounded, 1 = ignores context)
4. **Pronoun Resolution**: Were pronouns correctly resolved using conversation history? (5 = perfect resolution, 1 = wrong resolution, N/A if no pronouns)
5. **Factual Accuracy**: Based on the context, is the information factually correct? (5 = completely accurate, 1 = contains errors)
6. **Hallucination Detection**: Did the system make up information not in context? (5 = no hallucination, 1 = significant hallucination)

Provide your evaluation in this EXACT JSON format:
{{
    "retrieval_accuracy": {{"score": X, "explanation": "brief explanation"}},
    "answer_relevance": {{"score": X, "explanation": "brief explanation"}},
    "context_utilization": {{"score": X, "explanation": "brief explanation"}},
    "pronoun_resolution": {{"score": X, "explanation": "brief explanation or N/A"}},
    "factual_accuracy": {{"score": X, "explanation": "brief explanation"}},
    "hallucination_detection": {{"score": X, "explanation": "brief explanation"}},
    "overall_score": X.X,
    "summary": "brief overall assessment"
}}
"""

def get_judge_llm():
    """Get GPT-4 for evaluation"""
    return ChatOpenAI(
        model="gpt-4",
        temperature=0.0,
    )

def evaluate_response(query: str, context, response: str, chat_history: str = "") -> dict:
    """Use GPT-4 to evaluate the quality of a RAG response"""
    judge_llm = get_judge_llm()
    
    # Format context for evaluation
    if isinstance(context, list) and context:
        if hasattr(context[0], 'page_content'):
            formatted_context = "\n".join([f"- {doc.page_content}..." for doc in context[:5]])
        else:
            formatted_context = "\n".join([f"- {str(doc)}..." for doc in context[:5]])
    else:
        formatted_context = str(context)
    
    try:
        evaluation_prompt = JUDGE_PROMPT.format(
            query=query,
            context=formatted_context,
            response=response,
            chat_history=chat_history or "No previous context"
        )
        
        evaluation = judge_llm.invoke(evaluation_prompt)
        eval_content = evaluation.content.strip()
        
        # Extract JSON from response
        if "```json" in eval_content:
            start = eval_content.find("```json") + 7
            end = eval_content.find("```", start)
            eval_content = eval_content[start:end]
        elif "```" in eval_content:
            start = eval_content.find("```") + 3
            end = eval_content.find("```", start)
            eval_content = eval_content[start:end]
        
        return json.loads(eval_content)
        
    except json.JSONDecodeError as e:
        print(f"Failed to parse evaluation JSON: {e}")
        return {"error": "Could not parse evaluation", "overall_score": 0}
    except Exception as e:
        print(f"Evaluation error: {e}")
        return {"error": str(e), "overall_score": 0}

def test_with_evaluation():
    """Test the system and evaluate answers with GPT-4"""
    print("Testing system with GPT-4 evaluation...")
    print("=" * 50)
    
    retriever, rag_chain, total_docs = setup_rag_pipeline()
    if rag_chain is None:
        return
    
    # Simple test queries
    test_queries = [
        "Who is Tony Stark's daughter?",
        "What sacrifice did Iron Man make in Avengers: Endgame?", 
        "In what movies does Thanos appear?",
        "How is he defeated?"  # This should follow the Thanos question
    ]
    
    chat_context = ""
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        
        try:
            # Get response from RAG system
            result = rag_chain.invoke({"input": query, "chat_history": chat_history.messages})
            
            # Display retrieved documents
            if 'context' in result and result['context']:
                print(f"\nRetrieved {len(result['context'])} documents:")
                for j, doc in enumerate(result['context'][:5], 1):  # Show first 5 docs
                    title = doc.metadata.get('title', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
                    content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    print(f"  {j}. {title}: {content}...")
                print()
            
            print(f"Response: {result['answer']}")
            
            # Evaluate with GPT-4
            print("Evaluating with GPT-4...")
            evaluation = evaluate_response(
                query=query,
                context=result['context'],
                response=result['answer'],
                chat_history=chat_context
            )
            
            if "error" not in evaluation:
                print(f"Overall Score: {evaluation.get('overall_score', 'N/A')}/5")
                print(f"Summary: {evaluation.get('summary', 'No summary')}")
                
                # Show key metrics
                for metric in ['retrieval_accuracy', 'answer_relevance', 'context_utilization', 'pronoun_resolution']:
                    if metric in evaluation:
                        score_data = evaluation[metric]
                        if isinstance(score_data, dict):
                            score = score_data.get('score', 'N/A')
                            explanation = score_data.get('explanation', '')
                            print(f"  {metric.replace('_', ' ').title()}: {score}/5 - {explanation}")
            else:
                print(f"Evaluation failed: {evaluation.get('error', 'Unknown error')}")
            
            # Update chat context for next question
            chat_context += f"Q: {query}\nA: {result['answer']}\n\n"
            chat_history.add_user_message(query)
            chat_history.add_ai_message(result['answer'])
            
        except Exception as e:
            print(f"Error testing query '{query}': {e}")
        
        print("-" * 30)

if __name__ == "__main__":
    main()