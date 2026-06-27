# frontend/components/chat_message.py

import streamlit as st
from typing import Any


def _render_answer(content: str) -> None:
    """
    Renders the clean answer text returned by the backend.

    The backend (generator._post_process_answer) is responsible for
    extracting only the Answer section from the raw LLM output.
    The frontend receives a plain string and renders it directly.
    """
    st.markdown("#### 📝 Answer")
    st.markdown("---")
    st.markdown(content)


def _render_sources(sources: list[dict[str, Any]]) -> None:
    """
    Renders the list of retrieved source chunks as simplified, clean metadata.
    """
    if not sources:
        return

    st.markdown("---")
    st.markdown("#### 🔍 Sources")
    st.markdown("---")

    for i, source in enumerate(sources, start=1):
        document: str | None = source.get("document")
        section_path: str | None = source.get("section_path")
        section: str | None = source.get("section")
        page: int | None = source.get("page")

        lines: list[str] = []

        if document:
            lines.append(f"📄 {document}")

        path_to_show = section_path or section
        if path_to_show:
            lines.append(f"📍 {path_to_show}")

        if page is not None:
            lines.append(f"📄 Page {page}")

        if lines:
            source_block = "\n\n".join(lines)
            with st.container():
                st.markdown(source_block)
            if i < len(sources):
                st.markdown("")


def _render_evaluation(evaluation: dict[str, Any]) -> None:
    """
    Renders evaluation metrics from the structured EvaluationResponse dict.

    Only user-facing fields are shown:
        - Passed / Failed status
        - Grounding Score
        - Coverage Score
        - Hallucination Risk

    All values come directly from the structured backend response.
    No string manipulation is performed.
    """
    if not evaluation:
        return

    passed: bool | None = evaluation.get("passed")
    grounding_score: float | None = evaluation.get("grounding_score")
    coverage_score: float | None = evaluation.get("coverage_score")
    hallucination_risk: str | None = evaluation.get("hallucination_risk")

    has_evaluation = any(
        x is not None
        for x in [passed, grounding_score, coverage_score, hallucination_risk]
    )

    if not has_evaluation:
        return

    st.markdown("---")
    st.markdown("#### 📊 Evaluation")
    st.markdown("---")

    if passed is not None:
        status_label = "Passed ✅" if passed else "Failed ❌"
        st.markdown(f"**{status_label}**")

    if grounding_score is not None:
        st.markdown(f"**Grounding Score:** {grounding_score:.2f}")

    if coverage_score is not None:
        st.markdown(f"**Coverage Score:** {coverage_score:.2f}")

    if hallucination_risk is not None:
        risk_colour_map: dict[str, str] = {
            "Low": "green",
            "Medium": "orange",
            "High": "red",
        }
        colour = risk_colour_map.get(hallucination_risk, "gray")
        st.markdown(
            f"**Hallucination Risk:** "
            f"<span style='color:{colour}; font-weight:bold;'>"
            f"{hallucination_risk}</span>",
            unsafe_allow_html=True,
        )


def render_chat_message(message: dict[str, Any]) -> None:
    """
    Renders a single chat message.

    For user messages: displays the plain query text.

    For assistant messages: renders three structured sections —
        1. Answer      — clean answer text returned by the backend
        2. Sources     — structured list of retrieved source metadata
        3. Evaluation  — quality and safety metrics from the evaluator

    The frontend never parses LLM-generated markdown.  All data
    originates from structured fields in the ChatResponse schema.
    """
    role: str = message["role"]
    content: str = message["content"]

    if role == "user":
        st.chat_message("user").write(content)
        return

    with st.chat_message("assistant"):
        _render_answer(content)

        sources: list[dict[str, Any]] = message.get("sources") or []
        _render_sources(sources)

        evaluation: dict[str, Any] = message.get("evaluation") or {}
        _render_evaluation(evaluation)