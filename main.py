import asyncio
import logging
import os
from datetime import datetime, timedelta  # Add timedelta
from run_cycle import TradingCycle
from config import OPENROUTER_API_KEY

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.getLogger("langroid").setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    logger.handlers = []
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler("logs/terminal.log", mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logging.getLogger("langroid.utils.cache").setLevel(logging.WARNING)
    
    async def run_cycle_main():
        cycle = TradingCycle()
        while True:
            now = datetime.now()
            do_full_sweep = (now.hour == 0 and now.minute < 5 and 
                             (cycle.last_full_sweep is None or cycle.last_full_sweep.date() != now.date()))
            if do_full_sweep and cycle.full_sweep_count < 1:
                trigger_full = await cycle.short_cycle(full_search=True)
                cycle.full_sweep_count += 1
                cycle.last_full_sweep = now
            else:
                trigger_full = await cycle.short_cycle(full_search=False)
            if (not cycle.last_long_cycle or (now - cycle.last_long_cycle) >= timedelta(minutes=30)):
                await cycle.long_cycle()
                cycle.last_long_cycle = now
            if trigger_full and cycle.full_sweep_count < 2:
                logger.info("High VPH detected, triggering extra full sweep...")
                await cycle.short_cycle(full_search=True)
                cycle.full_sweep_count += 1
                cycle.last_full_sweep = now
            logger.info("Cycle complete, sleeping for 5 minutes...")
            await asyncio.sleep(300)  # 5 minutes
    
    asyncio.run(run_cycle_main())