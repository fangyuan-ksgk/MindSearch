import asyncio
import json
import logging
from copy import deepcopy
from dataclasses import asdict
from typing import Dict, List, Union
import os

import janus
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from lagent.schema import AgentStatusCode
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from mindsearch.agent import init_agent
from modal import Image, Stub, asgi_app, gpu, Secret, Mount

# Modal configuration
stub = Stub("mindsearch-api")

# Define the Modal image with all necessary dependencies
image = (
    Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install([
        "huggingface_hub", "torch", "tqdm", "fastapi", "uvicorn",
        "janus", "sse-starlette", "lagent", "pydantic"
    ])
    .apt_install("git", "git-lfs", "nodejs", "npm", "gcc")
    .run_commands(f"export HF_TOKEN={os.environ['HF_TOKEN']}")
    .run_commands(
        "git config --global user.name ksgk-fangyuan",
        "git config --global user.email fangyuan.yu18@gmail.com"
    )
    .run_commands(
        "cd /root && git clone https://github.com/InternLM/MindSearch && cd MindSearch &&"
        "pip install -r requirements.txt"
    )
)

# FastAPI app setup
web_app = FastAPI(docs_url='/')

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

class GenerationParams(BaseModel):
    inputs: Union[str, List[Dict]]
    agent_cfg: Dict = dict()

@web_app.get("/health")
async def health_check():
    return {"status": "healthy"}

@web_app.post('/solve')
async def run(request: GenerationParams):
    def convert_adjacency_to_tree(adjacency_input, root_name):
        def build_tree(node_name):
            node = {'name': node_name, 'children': []}
            if node_name in adjacency_input:
                for child in adjacency_input[node_name]:
                    child_node = build_tree(child['name'])
                    child_node['state'] = child['state']
                    child_node['id'] = child['id']
                    node['children'].append(child_node)
            return node
        return build_tree(root_name)

    async def generate():
        try:
            queue = janus.Queue()

            def sync_generator_wrapper():
                try:
                    for response in agent.stream_chat(inputs):
                        queue.sync_q.put(response)
                except Exception as e:
                    logging.exception(f'Exception in sync_generator_wrapper: {e}')
                finally:
                    queue.sync_q.put(None)

            async def async_generator_wrapper():
                loop = asyncio.get_event_loop()
                loop.run_in_executor(None, sync_generator_wrapper)
                while True:
                    response = await queue.async_q.get()
                    if response is None:
                        break
                    yield response
                    if not isinstance(response, tuple) and response.state == AgentStatusCode.END:
                        break

            async for response in async_generator_wrapper():
                if isinstance(response, tuple):
                    agent_return, node_name = response
                else:
                    agent_return = response
                    node_name = None
                origin_adj = deepcopy(agent_return.adjacency_list)
                adjacency_list = convert_adjacency_to_tree(agent_return.adjacency_list, 'root')
                assert adjacency_list['name'] == 'root' and 'children' in adjacency_list
                agent_return.adjacency_list = adjacency_list['children']
                agent_return = asdict(agent_return)
                agent_return['adj'] = origin_adj
                response_json = json.dumps(dict(response=agent_return, current_node=node_name), ensure_ascii=False)
                yield {'data': response_json}
        except Exception as exc:
            msg = 'An error occurred while generating the response.'
            logging.exception(msg)
            response_json = json.dumps(dict(error=dict(msg=msg, details=str(exc))), ensure_ascii=False)
            yield {'data': response_json}
        finally:
            queue.close()
            await queue.wait_closed()

    inputs = request.inputs
    agent = init_agent(lang='cn', model_format='internlm_server')
    return EventSourceResponse(generate())

@stub.function(
    image=image,
    gpu=gpu.A100(size="40GB"),
    secrets=[Secret.from_name("ksgk-secret")],
    allow_concurrent_inputs=100,
)
@asgi_app()
def fastapi_app():
    return web_app

# # This allows us to run the server locally if needed
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(web_app, host="0.0.0.0", port=8002)