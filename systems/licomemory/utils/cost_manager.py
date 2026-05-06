from typing import NamedTuple
from pydantic import BaseModel

from utils.token_counter import TOKEN_COSTS
from init.logger import logger


class Costs(NamedTuple):
    """Cost statistics container."""
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost: float
    total_budget: float


class CostManager(BaseModel):
    """Calculate and track the overhead of using LLM APIs."""

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_budget: float = 0
    max_budget: float = 100.0  # Default max budget $100
    total_cost: float = 0
    token_costs: dict[str, dict[str, float]] = TOKEN_COSTS
    stage_costs: list[Costs] = []

    def update_cost(self, prompt_tokens: int, completion_tokens: int, model: str):
        """
        Update the total cost, prompt tokens, and completion tokens.

        Args:
            prompt_tokens (int): The number of tokens used in the prompt.
            completion_tokens (int): The number of tokens used in the completion.
            model (str): The model used for the API call.
        """
        if prompt_tokens + completion_tokens == 0 or not model:
            return
        
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        
        if model not in self.token_costs:
            logger.warning(f"Model {model} not found in TOKEN_COSTS. Skipping cost calculation.")
            return

        cost = (
            prompt_tokens * self.token_costs[model]["prompt"]
            + completion_tokens * self.token_costs[model]["completion"]
        ) / 1000
        
        self.total_cost += cost

    def get_total_prompt_tokens(self) -> int:
        """Get the total number of prompt tokens."""
        return self.total_prompt_tokens

    def get_total_completion_tokens(self) -> int:
        """Get the total number of completion tokens."""
        return self.total_completion_tokens

    def get_total_cost(self) -> float:
        """Get the total cost of API calls."""
        return self.total_cost

    def get_costs(self) -> Costs:
        """Get all costs."""
        return Costs(
            self.total_prompt_tokens, 
            self.total_completion_tokens, 
            self.total_cost, 
            self.total_budget
        )

    def set_stage_cost(self):
        """Set the cost of the current stage."""
        self.stage_costs.append(self.get_costs())

    def get_last_stage_cost(self) -> Costs:
        """Get the cost of the last stage."""
        current_cost = self.get_costs()
        
        if len(self.stage_costs) == 0:
            last_cost = Costs(0, 0, 0, 0)
        else:
            last_cost = self.stage_costs[-1]
        
        last_stage_cost = Costs(
            current_cost.total_prompt_tokens - last_cost.total_prompt_tokens,
            current_cost.total_completion_tokens - last_cost.total_completion_tokens,
            current_cost.total_cost - last_cost.total_cost,
            current_cost.total_budget - last_cost.total_budget
        )
        
        self.set_stage_cost()
        return last_stage_cost

    def check_budget(self) -> bool:
        """Check if total cost exceeds budget."""
        if self.total_cost > self.max_budget:
            logger.error(f"❌ Budget exceeded! Total cost: ${self.total_cost:.4f} > Max budget: ${self.max_budget:.2f}")
            return False
        elif self.total_cost > self.max_budget * 0.8:
            logger.warning(f"⚠️ Budget warning! Total cost: ${self.total_cost:.4f} is 80%+ of max budget: ${self.max_budget:.2f}")
        return True

    def get_cost_totals(self) -> dict:
        """Get aggregate prompt/completion/cost totals."""
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "max_budget_usd": self.max_budget,
            "budget_usage_percent": round((self.total_cost / self.max_budget) * 100, 2) if self.max_budget > 0 else 0,
            "stages_tracked": len(self.stage_costs)
        }


class TokenCostManager(CostManager):
    """For self-hosted models that are free and without cost."""

    def update_cost(self, prompt_tokens: int, completion_tokens: int, model: str):
        """Update tokens without cost calculation for self-hosted models."""
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens


class GraphBuildingCostManager(CostManager):
    """Specialized cost manager for graph building operations."""
    
    chunking_tokens: int = 0
    entity_extraction_tokens: int = 0
    relationship_extraction_tokens: int = 0
    graph_construction_tokens: int = 0
    summary_generation_tokens: int = 0
        
    def update_chunking_cost(self, prompt_tokens: int, completion_tokens: int, model: str):
        """Update cost for chunking operations."""
        self.chunking_tokens += prompt_tokens + completion_tokens
        self.update_cost(prompt_tokens, completion_tokens, model)
        logger.info(f"📄 Chunking Cost - Prompt: {prompt_tokens}, Completion: {completion_tokens}")
    
    def update_entity_extraction_cost(self, prompt_tokens: int, completion_tokens: int, model: str):
        """Update cost for entity extraction operations."""
        self.entity_extraction_tokens += prompt_tokens + completion_tokens
        self.update_cost(prompt_tokens, completion_tokens, model)
    
    def update_relationship_extraction_cost(self, prompt_tokens: int, completion_tokens: int, model: str):
        """Update cost for relationship extraction operations."""
        self.relationship_extraction_tokens += prompt_tokens + completion_tokens
        self.update_cost(prompt_tokens, completion_tokens, model)
        logger.info(f"🔗 Relationship Extraction Cost - Prompt: {prompt_tokens}, Completion: {completion_tokens}")
    
    def update_graph_construction_cost(self, prompt_tokens: int, completion_tokens: int, model: str):
        """Update cost for graph construction operations."""
        self.graph_construction_tokens += prompt_tokens + completion_tokens
        self.update_cost(prompt_tokens, completion_tokens, model)
        logger.info(f"🕸️ Graph Construction Cost - Prompt: {prompt_tokens}, Completion: {completion_tokens}")
    
    def update_summary_generation_cost(self, prompt_tokens: int, completion_tokens: int, model: str):
        """Update cost for summary generation operations."""
        self.summary_generation_tokens += prompt_tokens + completion_tokens
        self.update_cost(prompt_tokens, completion_tokens, model)
    
    def get_graph_building_summary(self) -> dict:
        """Get detailed cost summary for graph building."""
        summary = self.get_cost_totals()
        summary.update({
            "chunking_tokens": self.chunking_tokens,
            "entity_extraction_tokens": self.entity_extraction_tokens,
            "relationship_extraction_tokens": self.relationship_extraction_tokens,
            "graph_construction_tokens": self.graph_construction_tokens,
            "summary_generation_tokens": self.summary_generation_tokens,
            "total_graph_building_tokens": (
                self.chunking_tokens + 
                self.entity_extraction_tokens + 
                self.relationship_extraction_tokens + 
                self.graph_construction_tokens +
                self.summary_generation_tokens
            )
        })
        return summary


class QueryCostManager(CostManager):
    """Specialized cost manager for query operations."""
    
    search_stage_tokens: int = 0
    response_stage_tokens: int = 0
        
    def update_retrieval_cost(self, prompt_tokens: int, completion_tokens: int, model: str):
        """Update cost for retrieval operations."""
        self.search_stage_tokens += prompt_tokens + completion_tokens
        self.update_cost(prompt_tokens, completion_tokens, model)
        logger.info(f"🔍 Retrieval Cost - Prompt: {prompt_tokens}, Completion: {completion_tokens}")
    
    def update_answer_generation_cost(self, prompt_tokens: int, completion_tokens: int, model: str):
        """Update cost for answer generation operations."""
        self.response_stage_tokens += prompt_tokens + completion_tokens
        self.update_cost(prompt_tokens, completion_tokens, model)
        logger.info(f"💬 Answer Generation Cost - Prompt: {prompt_tokens}, Completion: {completion_tokens}")
    
    def get_runtime_cost_totals(self) -> dict:
        """Get MemOS-aligned runtime cost totals for query operations."""
        summary = self.get_cost_totals()
        summary.update({
            "search_stage_tokens": self.search_stage_tokens,
            "response_stage_tokens": self.response_stage_tokens,
        })
        return summary
