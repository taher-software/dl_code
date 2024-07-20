from src.config import Config
import logging

import websockets

async def socket_broadcast(message, qParams = ''):
    try:
        async with websockets.connect(f"{Config.SOCKET_URI}{qParams}") as websocket:
            await websocket.send(message.encode('utf-8'))
    except Exception as err: 
        logging.info(f"socket_broadcast  - {Config.SOCKET_URI}{qParams}")
        logging.error('unable to broadcast', err)
          