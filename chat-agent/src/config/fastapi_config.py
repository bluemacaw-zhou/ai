"""Class-based FastAPI application configuration."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from ai_pad_bootstrap import AiPadBootstrap
from controller.chat_controller import router as chat_router


class FastApiConfig:
    """Creates and configures the application's FastAPI instance."""

    @staticmethod
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.main_graph = await AiPadBootstrap.build_main_graph()
        yield

    @classmethod
    def create_app(cls) -> FastAPI:
        app = FastAPI(title="Chat Agent", lifespan=cls.lifespan)
        app.include_router(chat_router)
        return app


app = FastApiConfig.create_app()
