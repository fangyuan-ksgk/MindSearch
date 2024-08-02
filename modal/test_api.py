# from pydantic import BaseModel
# from fastapi.responses import HTMLResponse

# from modal import Image, App, web_endpoint

# image = Image.debian_slim().pip_install("boto3")
# app = App(image=image)


# class Item(BaseModel):
#     name: str
#     qty: int = 42


# @app.function()
# @web_endpoint(method="POST")
# def f(item: Item):
#     import boto3
#     # do things with boto3...
#     return HTMLResponse(f"<html>Hello, {item.name}!</html>")


from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from modal import Image, App, asgi_app

web_app = FastAPI()
app = App()

image = Image.debian_slim().pip_install("boto3")


@web_app.post("/foo")
async def foo(request: Request):
    body = await request.json()
    return body


@web_app.get("/bar")
async def bar(arg="world"):
    return HTMLResponse(f"<h1>Hello Fast {arg}!</h1>")


@app.function(image=image)
@asgi_app()
def fastapi_app():
    return web_app