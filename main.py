from fastapi import FastAPI, Request, Response, BackgroundTasks
import gradio as gr
import uvicorn
import os
import time
import logging
from components.setup_interface import setup_interface
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define function to check if setup is necessary
def is_setup_needed():
    # Check if the setup.flag file exists
    return not os.path.exists("setup.flag")

# Create FastAPI app instance
app = FastAPI()

# Middleware to check if setup is necessary
@app.middleware("http")
async def check_setup_needed(request: Request, call_next):
    logger.info(f"Received request: {request.url.path}")
    if not is_setup_needed() and request.url.path.startswith("/setup"):
        # If setup is not needed, forward requests from /setup to /main
        logger.info("Setup not needed, redirecting to /main")
        return Response(status_code=307, headers={"Location": "/main"})
    response = await call_next(request)
    return response

# Mount Gradio Setup Screen at "/setup"
app = gr.mount_gradio_app(app, setup_interface, path="/setup")

# Function to monitor the setup.flag file and reload the app
async def monitor_setup_flag():
    while is_setup_needed():
        await asyncio.sleep(1)
    logger.info("Setup complete, mounting main interface...")
    from components.main_interface import main_interface_blocks
    app.mount("/main", gr.mount_gradio_app(app, main_interface_blocks, path="/main"))

# Serve PDF files dynamically through FastAPI
@app.get("/pdf")
async def get_pdf(path: str):
    if os.path.exists(path):
        return Response(content=open(path, 'rb').read(), media_type='application/pdf')
    else:
        return Response(status_code=404, content="PDF not found")

# Define root route to handle redirection after setup
@app.get("/")
async def root(background_tasks: BackgroundTasks):
    if is_setup_needed():
        logger.info("Setup needed, redirecting to /setup")
        background_tasks.add_task(monitor_setup_flag)
        return Response(status_code=307, headers={"Location": "/setup"})
    logger.info("Setup not needed, redirecting to /main")
    return Response(status_code=307, headers={"Location": "/main"})

if __name__ == "__main__":
    if is_setup_needed():
        logger.info("Starting server in setup mode...")
    else:
        from components.main_interface import main_interface_blocks
        app = gr.mount_gradio_app(app, main_interface_blocks, path="/main")
        logger.info("Starting server in main mode...")
    uvicorn.run(app, host="127.0.0.1", port=7860)