"""Entry point for zero-dte-scalper."""

import asyncio
import signal
import sys

from src.core.engine import TradingEngine
from src.infra.config import get_settings
from src.infra.logger import setup_logging


def main() -> None:
    settings = get_settings()
    setup_logging()

    engine = TradingEngine(settings)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def shutdown(sig, frame):
        loop.create_task(engine.stop())

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        loop.run_until_complete(engine.start())
    except KeyboardInterrupt:
        loop.run_until_complete(engine.stop())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
