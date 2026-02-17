import ray
from ray import serve

from dlk.ray_server import build_app

if __name__ == "__main__":
    # 1. Initialize Ray cluster locally
    ray.init()

    # 2. Build the app from configuration
    app = build_app("./config/serve.jsonc")

    # 3. Deploy the application to Ray Serve
    # bind to 0.0.0.0:8000
    serve.run(app, host="0.0.0.0", port=8000)

    print("Ray Serve is running. Press Ctrl+C to terminate.")

    # Keep the main thread alive
    import time

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
