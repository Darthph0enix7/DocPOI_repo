from fastapi import FastAPI, Request, Response
import gradio as gr
import uvicorn
import os
from setup_interface import setup_interface
from main_interface import main_interface_blocks
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
    if not is_setup_needed() and request.url.path.startswith("/setup"):
        # If setup is not needed, forward requests from /setup to /main
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
        return Response(status_code=307, headers={"Location": "/setup"})
    return Response(status_code=307, headers={"Location": "/main"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)