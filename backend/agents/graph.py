# backend/agents/graph.py
import logging
from typing import Literal,cast

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from backend.agents.state import EnterpriseState
from backend.agents.nodes import (
    route_node,
    document_search_node,
    web_search_node,
    hybrid_search_node,
    generate_answer_node,
    evaluate_answer_node,
)

logger = logging.getLogger(__name__)

__all__ = [
    "build_graph",
    "get_graph",
]

_ALLOWED_ROUTES = frozenset(
    {
        "documents",
        "web",
        "hybrid",
    }
)


def _route_condition(state: EnterpriseState) -> Literal["documents", "web", "hybrid"]:
    """
    Determines the next node to execute based on the routing decision computed by the router.

    Args:
        state: The current enterprise workflow state.

    Returns:
        The Literal name of the next node's path: 'documents', 'web', or 'hybrid'.
    """
    route = state.get("route")
    allowed_routes = _ALLOWED_ROUTES

    if route not in allowed_routes:
        logger.warning(
            "Invalid or missing route '%s' in state. Falling back to 'documents'.",
            route,
        )
        return "documents"

    return cast(Literal["documents", "web", "hybrid"], route)


def build_graph() -> CompiledStateGraph:
    """
    Constructs, links, and compiles the LangGraph StateGraph workflow.

    Returns:
        CompiledStateGraph: The compiled state machine ready for execution.
    """
    logger.info("Graph build started.")

    # 1. Initialize StateGraph with the schema
    builder: StateGraph = StateGraph(EnterpriseState)
    
    logger.info("Compiling LangGraph workflow.")

    # 2. Add node functions to the graph
    builder.add_node("route_node", route_node)
    builder.add_node("document_search_node", document_search_node)
    builder.add_node("web_search_node", web_search_node)
    builder.add_node("hybrid_search_node", hybrid_search_node)
    builder.add_node("generate_answer_node", generate_answer_node)
    builder.add_node("evaluate_answer_node", evaluate_answer_node)

    # 3. Add edges and define execution transitions
    builder.add_edge(START, "route_node")

    # Add the router node's conditional outbound edges
    builder.add_conditional_edges(
        "route_node",
        _route_condition,
        {
            "documents": "document_search_node",
            "web": "web_search_node",
            "hybrid": "hybrid_search_node",
        },
    )

    # Connect all search node outputs to the generation node
    builder.add_edge("document_search_node", "generate_answer_node")
    builder.add_edge("web_search_node", "generate_answer_node")
    builder.add_edge("hybrid_search_node", "generate_answer_node")

    # Connect answer generation to response quality evaluation
    builder.add_edge("generate_answer_node", "evaluate_answer_node")
    builder.add_edge("evaluate_answer_node", END)

    # 4. Compile the state machine workflow
    compiled_graph = builder.compile()

    logger.info("Graph build completed.")
    return compiled_graph


# lazy-initialized compiled graph singleton
_GRAPH: CompiledStateGraph | None = None


def get_graph() -> CompiledStateGraph:
    """
    Returns the lazy-initialized singleton instance of the compiled StateGraph workflow.
    """
    global _GRAPH
    if _GRAPH is None:
        logger.info("Initializing LangGraph singleton.")
        _GRAPH = build_graph()

    return _GRAPH