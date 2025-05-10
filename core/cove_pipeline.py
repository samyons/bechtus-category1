import os
import logging
import re
from typing import Dict, Any, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.schema import Document

from .llm_prompts import (
    get_initial_rag_prompt,
    get_verification_planning_prompt,
    get_verification_answering_prompt,
    get_final_revision_prompt
)

logger = logging.getLogger(__name__)

LLM_MODEL_NAME = "gemini-2.0-flash"
LLM_TEMPERATURE_DRAFT = 0.3
LLM_TEMPERATURE_VERIFICATION_PLAN = 0.1 # Planning VQs needs some creativity but also precision
LLM_TEMPERATURE_VERIFICATION_ANSWER = 0.0 # Highly factual for answering VQs
LLM_TEMPERATURE_REVISION = 0.15 # Factual but coherent revision

def get_llm(temperature: float) -> Optional[ChatGoogleGenerativeAI]:
    """Initializes and returns the Gemini LLM instance with specified temperature."""
    try:
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            temperature=temperature,
            convert_system_message_to_human=True
            # top_p=0.95, # Experiment with top_p if needed
            # top_k=40    # Experiment with top_k if needed
        )
        logger.debug(f"LLM instance created: model={LLM_MODEL_NAME}, temp={temperature}")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM ({LLM_MODEL_NAME}): {e}", exc_info=True)
        return None

def _format_docs(docs: List[Document]) -> str:
    """Formats retrieved documents into a single string for the prompt context."""
    if not docs:
        return "No relevant context found in AAOIFI standards for this query."
    
    unique_docs_content = {}
    for doc in docs:
        content_preview = doc.page_content[:50] # Use a small part of content for uniqueness
        # Key includes source, page, and content preview to handle similar content on different pages or from different sources slightly differently.
        unique_key = (doc.metadata.get('source', 'N/A'), str(doc.metadata.get('page', 'N/A')), content_preview)
        
        if unique_key not in unique_docs_content:
            source_name = os.path.basename(doc.metadata.get('source', 'N/A')) if doc.metadata.get('source') else 'N/A'
            page_number = doc.metadata.get('page', 'N/A') if doc.metadata.get('page') is not None else 'N/A'
            unique_docs_content[unique_key] = f"Source: {source_name}, Page: {page_number}\nContent:\n{doc.page_content}"

    return "\n\n---\n\n".join(unique_docs_content.values())


def _parse_vqs(llm_output: str) -> List[str]:
    """Parses lines starting with 'VQ: ' from LLM output."""
    questions = []
    for line in llm_output.splitlines():
        if line.strip().startswith("VQ:"):
            question_text = line.strip()[3:].strip()
            if question_text: # Add only if not empty after stripping "VQ: "
                questions.append(question_text)
    return questions

class CoVeChainOrchestrator:
    def __init__(self, retriever: VectorStoreRetriever):
        self.retriever = retriever
        self.llm_draft = get_llm(LLM_TEMPERATURE_DRAFT)
        self.llm_v_plan = get_llm(LLM_TEMPERATURE_VERIFICATION_PLAN)
        self.llm_v_answer = get_llm(LLM_TEMPERATURE_VERIFICATION_ANSWER)
        self.llm_revise = get_llm(LLM_TEMPERATURE_REVISION)

        if not all([self.llm_draft, self.llm_v_plan, self.llm_v_answer, self.llm_revise]):
            raise ValueError("One or more LLM instances could not be initialized. Check API key and model availability.")

        self.prompt_initial = get_initial_rag_prompt()
        self.prompt_v_plan = get_verification_planning_prompt()
        self.prompt_v_answer = get_verification_answering_prompt()
        self.prompt_revise = get_final_revision_prompt()

        # Chains
        self.chain_draft_answer = self.prompt_initial | self.llm_draft | StrOutputParser()
        self.chain_verification_planning = self.prompt_v_plan | self.llm_v_plan | StrOutputParser()
        self.chain_verification_answering = self.prompt_v_answer | self.llm_v_answer | StrOutputParser()
        self.chain_final_revision = self.prompt_revise | self.llm_revise | StrOutputParser()


    def _retrieve_context_for_query(self, query: str) -> str:
        """Retrieves and formats context for a given query."""
        docs = self.retriever.invoke(query)
        return _format_docs(docs)

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the full Chain-of-Verification pipeline.
        Input: {"scenario": str, "question": str}
        Output: {"final_answer": str, "draft_answer": str, "verification_log": List[Dict]}
        """
        scenario = inputs["scenario"]
        question = inputs["question"]
        logger.info(f"\n--- Starting CoVe Process for Question: {question[:70]}... ---")

        # 1. Generate Initial Context and Draft Answer
        logger.info("Step 1: Generating initial context and draft answer...")
        initial_context_query = f"Scenario: {scenario}\nQuestion: {question}"
        initial_context = self._retrieve_context_for_query(initial_context_query)

        if "No relevant context found" in initial_context:
            logger.warning("CoVe: No initial context retrieved. CoVe cannot proceed effectively.")
            return {
                "final_answer": "Could not generate an answer due to lack of relevant context in the AAOIFI standards for the query.",
                "draft_answer": "N/A - No context retrieved.",
                "verification_log": []
            }

        draft_answer = self.chain_draft_answer.invoke({
            "context": initial_context, "scenario": scenario, "question": question
        })
        logger.info(f"CoVe: Draft Answer generated (length: {len(draft_answer)}).")
        # logger.debug(f"CoVe Draft Answer Content:\n{draft_answer}")


        # 2. Plan Verification Questions
        logger.info("Step 2: Planning verification questions...")
        verification_questions_str = self.chain_verification_planning.invoke({
            "draft_answer": draft_answer, "original_question": question
        })
        verification_questions_list = _parse_vqs(verification_questions_str)
        logger.info(f"CoVe: Planned {len(verification_questions_list)} Verification Questions.")
        # logger.debug(f"CoVe Planned VQs:\n{verification_questions_list}")


        if not verification_questions_list:
            logger.info("CoVe: No verification questions planned. Using draft answer as final.")
            return {
                "final_answer": draft_answer,
                "draft_answer": draft_answer,
                "verification_log": [{"info": "No verification questions were planned by the LLM."}]
            }

        # 3. Answer Verification Questions (with targeted retrieval)
        logger.info("Step 3: Answering verification questions with targeted retrieval...")
        verification_log = []
        for vq in verification_questions_list:
            logger.debug(f"CoVe: Processing VQ: {vq}")
            # TARGETED RETRIEVAL FOR EACH VERIFICATION QUESTION:
            vq_specific_context = self._retrieve_context_for_query(vq) # Use VQ itself to get specific context
            
            context_for_vq_answer = vq_specific_context
            context_source_log = "Targeted VQ Context"
            if "No relevant context found" in vq_specific_context:
                logger.warning(f"CoVe: No specific context found for VQ: '{vq}'. Falling back to initial context.")
                context_for_vq_answer = initial_context # Fallback to broader context
                context_source_log = "Initial Context (fallback)"

            vq_answer = self.chain_verification_answering.invoke({
                "verification_question": vq, "context": context_for_vq_answer
            })
            verification_log.append({"question": vq, "answer": vq_answer, "context_source": context_source_log})
            logger.debug(f"CoVe VQ Answered: Q: {vq[:60]}... A: {vq_answer[:60]}...")
        
        verification_summary_str = "\n\n".join(
            [f"Verification Question: {item['question']}\nAnswer based on context ({item['context_source']}):\n{item['answer']}" for item in verification_log]
        )

        # 4. Revise Final Answer
        logger.info("Step 4: Revising and generating final answer...")
        final_answer = self.chain_final_revision.invoke({
            "original_question": question, "original_scenario": scenario,
            "draft_answer": draft_answer, "verification_summary": verification_summary_str,
            "initial_context": initial_context # Provide initial context again for reference during revision
        })
        logger.info(f"CoVe: Final Verified Answer generated (length: {len(final_answer)}).")
        # logger.debug(f"CoVe Final Answer Content:\n{final_answer}")

        return {
            "final_answer": final_answer,
            "draft_answer": draft_answer,
            "verification_log": verification_log
        }