from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.dependencies import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import setup_logging
from app.routers import health, pdf, webhook


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    setup_logging(settings)
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    # Start the audit poller as a background thread when audit_mode is
    # "poll" or "both". In "webhook" mode (default) this is a no-op —
    # the FastAPI server only listens for incoming webhooks.
    from app.services.audit.poller import start_poller_thread, stop_poller_thread
    start_poller_thread()
    try:
        yield
    finally:
        stop_poller_thread()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://idp-frontend-nu.vercel.app",
            "http://localhost:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(pdf.router)
    app.include_router(webhook.router)

    return app


app = create_app()
