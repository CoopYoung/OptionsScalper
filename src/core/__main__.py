"""Entry point for zero-dte-scalper."""

import asyncio
import logging
import signal

from src.core.engine import TradingEngine
from src.infra.config import get_settings
from src.infra.logger import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    setup_logging()

    engine = TradingEngine(settings)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    shutdown_triggered = False

    def shutdown(sig, frame):
        nonlocal shutdown_triggered
        sig_name = signal.Signals(sig).name
        if shutdown_triggered:
            logger.warning("Second %s received — forcing exit (positions persisted)", sig_name)
            raise SystemExit(1)
        shutdown_triggered = True
        logger.info("Received %s — initiating graceful shutdown...", sig_name)
        loop.create_task(engine.stop())

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        loop.run_until_complete(engine.start())
    except KeyboardInterrupt:
        if not shutdown_triggered:
            loop.run_until_complete(engine.stop())
    except SystemExit:
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()
