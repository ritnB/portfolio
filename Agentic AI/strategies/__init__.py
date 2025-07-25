# strategies/__init__.py
from .content_generator import ContentGenerator, generate_targeted_content
from .promotion_strategy import PromotionStrategy

__all__ = ["ContentGenerator", "generate_targeted_content", "PromotionStrategy"] 