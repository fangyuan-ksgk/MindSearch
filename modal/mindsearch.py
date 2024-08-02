import modal
from modal import Image, App, gpu, web_endpoint
from huggingface_hub import login
import os
import subprocess

HF_TOKEN = os.environ["HF_TOKEN"]
login(HF_TOKEN)


image = (
    Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install(
        ["huggingface_hub", "torch", "tqdm", "fastapi", "uvicorn"]
    )
    .apt_install("git", "git-lfs", "nodejs", "npm", "gcc")
    .run_commands(f"export HF_TOKEN={HF_TOKEN}")
    .run_commands("git config --global user.name ksgk-fangyuan",
                  "git config --global user.email fangyuan.yu18@gmail.com")
    .run_commands(
        "cd /root && git clone https://github.com/InternLM/MindSearch && cd MindSearch &&"
        "pip install -r requirements.txt"
    )
)

app = App(name="mindsearch-app", image=image)


# ## Mounting the `app.py` script
#
# We can just mount the `app.py` script inside the container at a pre-defined path using a Modal
# [`Mount`](https://modal.com/docs/guide/local-data#mounting-directories).

import shlex
import subprocess
from pathlib import Path

streamlit_script_local_path = Path(__file__).parent / "mindsearch_app.py"
streamlit_script_remote_path = Path("/root/mindsearch_app.py")

if not streamlit_script_local_path.exists():
    raise RuntimeError(
        "app.py not found! Place the script with your streamlit app in the same directory."
    )

streamlit_script_mount = modal.Mount.from_local_file(
    streamlit_script_local_path,
    streamlit_script_remote_path,
)

@app.function(
    image=image,
    gpu=gpu.A100(size="40GB"),
    secrets=[modal.Secret.from_name("ksgk-secret")],
    allow_concurrent_inputs=100,
    mounts=[streamlit_script_mount]
)
@modal.web_server(8000)
def run():

    # Start MindSearch server
    subprocess.Popen(
        "cd /root/MindSearch && python -m mindsearch.app --lang en --model_format internlm_server",
        shell=True
    )
    
    # Start Streamlit app
    target = shlex.quote(str(streamlit_script_remote_path))
    cmd = f"streamlit run {target} --server.port 8000 --server.enableCORS=false --server.enableXsrfProtection=false"
    subprocess.Popen(cmd, shell=True)



# ## Iterate and Deploy
#
# While you're iterating on your screamlit app, you can run it "ephemerally" with `modal serve`. This will
# run a local process that watches your files and updates the app if anything changes.
#
# ```shell
# modal serve serve_streamlit.py
# ```
#
# Once you're happy with your changes, you can deploy your application with
#
# ```shell
# modal deploy serve_streamlit.py
# ```
#
# If successful, this will print a URL for your app, that you can navigate to from
# your browser 🎉 .