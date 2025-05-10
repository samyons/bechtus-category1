# aaoifi_cove_streamlit_app/core/llm_prompts.py
from langchain.prompts import PromptTemplate

def get_initial_rag_prompt() -> PromptTemplate:
    """Prompt for generating the initial draft answer."""
    return PromptTemplate(
        input_variables=["context", "scenario", "question"],
        template="""
        You are an Islamic Finance Expert. Based on the **Provided AAOIFI Standards Context** ONLY,
        generate a comprehensive draft answer to the **Original Question** related to the **Original Use Case Scenario**.
        Your draft should attempt to be accurate and cover key aspects. Cite sources from the context (e.g., FAS X, para Y).

        **Provided AAOIFI Standards Context:**
        --------------------
        {context}
        --------------------

        **Original Use Case Scenario:**
        --------------------
        {scenario}
        --------------------

        **Original Question:**
        --------------------
        {question}
        --------------------

        **Draft Answer:**
        """
    )

def get_verification_planning_prompt() -> PromptTemplate:
    """Prompt for generating verification questions based on a draft answer."""
    return PromptTemplate(
        input_variables=["draft_answer", "original_question"],
        template="""
        You are a meticulous Shariah Auditor. Review the following **Draft Answer** to the **Original Question**.
        Your task is to formulate 2-4 specific, targeted verification questions that would help fact-check the key claims, calculations, or interpretations made in the draft answer.
        These questions should be answerable by referring back to AAOIFI standards.
        List each verification question on a new line, starting with "VQ: ".
        Focus on questions that challenge assumptions or require specific standard citations.

        **Original Question:**
        --------------------
        {original_question}
        --------------------

        **Draft Answer to Verify:**
        --------------------
        {draft_answer}
        --------------------

        **Verification Questions:**
        """
    )

def get_verification_answering_prompt() -> PromptTemplate:
    """Prompt for answering a single verification question using provided context."""
    return PromptTemplate(
        input_variables=["verification_question", "context"],
        template="""
        You are an AI assistant. Based ONLY on the **Provided AAOIFI Standards Context**,
        answer the following **Verification Question** factually and concisely.
        If the context does not contain the answer, state "Information not found in the provided context for this specific verification question."
        Cite the specific source (e.g., document name, page number, clause) from the context if possible.

        **Provided AAOIFI Standards Context:**
        --------------------
        {context}
        --------------------

        **Verification Question:**
        --------------------
        {verification_question}
        --------------------

        **Answer to Verification Question:**
        """
    )

def get_final_revision_prompt() -> PromptTemplate:
    """Prompt for revising the draft answer based on verification Q&A."""
    return PromptTemplate(
        input_variables=["original_question", "original_scenario", "draft_answer", "verification_summary", "initial_context"],
        template="""
        You are an expert Islamic Finance writer tasked with producing a final, verified answer.
        You have an **Original Question** about an **Original Scenario**, an initial **Draft Answer**,
        a **Summary of Verification Questions and their Answers**, and the **Initial AAOIFI Context** used for the draft.

        Your goal is to critically revise the **Draft Answer** to be more accurate, well-supported, and comprehensive, incorporating insights from the **Verification Summary**.
        - Ensure all claims in the final answer are consistent with the verification findings.
        - If a claim in the draft was found to be incorrect, unsubstantiated, or imprecise by the verification, correct it thoroughly or remove it if unfixable.
        - If new relevant details or nuances emerged from verification, integrate them to enrich the answer.
        - The final answer must be based ONLY on the **Initial AAOIFI Context** and the **Verification Summary**. Do not introduce new external information.
        - Clearly cite sources (e.g., FAS X, para Y from the Initial Context) for key statements and calculations in your final answer.
        - If calculations were part of the original question, ensure they are re-checked and accurately presented based on verified principles.
        - Structure the final answer professionally, with clear explanations. Use journal entry format (Dr./Cr.) if accounting entries are requested.

        **Original Question:**
        --------------------
        {original_question}
        --------------------

        **Original Scenario:**
        --------------------
        {original_scenario}
        --------------------

        **Initial AAOIFI Context (used for draft and verification lookup):**
        --------------------
        {initial_context}
        --------------------

        **Draft Answer:**
        --------------------
        {draft_answer}
        --------------------

        **Summary of Verification Questions and Their Answers (use these to guide your revision):**
        --------------------
        {verification_summary}
        --------------------

        **Revised and Verified Final Answer:**
        """
    )