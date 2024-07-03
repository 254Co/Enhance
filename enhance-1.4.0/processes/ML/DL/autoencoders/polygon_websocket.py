# polygon_websocket.py
import websocket
import json

def start_polygon_stream(api_key, process_batch_callback):
    def on_message(ws, message):
        data = json.loads(message)
        process_batch_callback(data)

    def on_error(ws, error):
        print(error)

    def on_close(ws):
        print("### closed ###")

    def on_open(ws):
        ws.send(json.dumps({"action": "auth", "params": api_key}))
        ws.send(json.dumps({"action": "subscribe", "params": "T.*"}))

    websocket.enableTrace(True)
    ws = websocket.WebSocketApp("wss://socket.polygon.io/stocks",
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()
