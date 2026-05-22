from app.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore


class Retriever:
    @staticmethod
    def retrieve(query: str, kb_id: str,
                 top_k_text: int | None = None,
                 top_k_image: int | None = None
                 ) -> tuple[list[dict], list[dict]]:
        top_k_text = top_k_text or settings.retrieve_top_k_text
        top_k_image = top_k_image or settings.retrieve_top_k_image

        text_embedding = EmbeddingService.embed_text_bge(query)
        image_embedding = EmbeddingService.embed_text_clip(query)

        text_results = VectorStore.search_text(text_embedding, kb_id, top_k_text)
        image_results = VectorStore.search_image(image_embedding, kb_id, top_k_image)

        return text_results, image_results
