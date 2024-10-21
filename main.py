from fastapi import FastAPI, Request, Response
import gradio as gr
import uvicorn
import os
from setup_interface import setup_interface
from main_interface import main_interface_blocks
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define function to check if setup is necessary
def is_setup_needed():
    # Check if the setup.flag file exists
    return not os.path.exists("setup.flag")

# Define Main Screen Function
def main_interface():
    return "Welcome to the Main Screen!"

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

# Mount Gradio Main Screen at "/main"
app = gr.mount_gradio_app(app, main_interface_blocks, path="/main")

# Define root route to handle redirection after setup
@app.get("/")
async def root():
    if is_setup_needed():
        logger.info("Setup needed, redirecting to /setup")
        return Response(status_code=307, headers={"Location": "/setup"})
    logger.info("Setup not needed, redirecting to /main")
    return Response(status_code=307, headers={"Location": "/main"})

if __name__ == "__main__":
    logger.info("Starting server...")
    uvicorn.run(app, host="127.0.0.1", port=7860)