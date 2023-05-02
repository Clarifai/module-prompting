import uuid

import streamlit as st
from clarifai.auth.helper import ClarifaiAuthHelper
from clarifai.client import create_stub
from clarifai.listing.lister import ClarifaiResourceLister
from clarifai.modules.css import ClarifaiStreamlitCSS
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf import json_format

ClarifaiStreamlitCSS.insert_default_css(st)

DEBUG = False

COHERE = "cohere: generate-base"
OPENAI = "openai: gpt-3.5-turbo"
AI21_A = "ai21: j2-jumbo-instruct"
AI21_B = "ai21: j2-grande-instruct"
AI21_C = "ai21: j2-jumbo"
AI21_D = "ai21: j2-grande"
AI21_E = "ai21: j2-large"

API_INFO = {
    COHERE: {
        "user_id": "cohere",
        "app_id": "generate",
        "model_id": "generate-base",
        "version_id": "07bf79a08a45492d8be5c49085244f1c"
    },
    OPENAI: {
        "user_id": "openai",
        "app_id": "chat_completion",
        "model_id": "chatgpt-3_5-turbo",
        "version_id": "8312408ae32f40cd9322804accf17d50",
    },
    AI21_A: {
        "user_id": "ai21",
        "app_id": "complete",
        "model_id": "j2-jumbo-instruct",
        "version_id": "2ca88c8e94e14b02bb20c39bc74ffbfe"
    },
    AI21_B: {
        "user_id": "ai21",
        "app_id": "complete",
        "model_id": "j2-grande-instruct",
        "version_id": "2910ac4666e444e79ea70bb960a293ef"
    },
    AI21_C: {
        "user_id": "ai21",
        "app_id": "complete",
        "model_id": "j2-jumbo",
        "version_id": "80dbb77f961e4840a84bea93ac19c242"
    },
    AI21_D: {
        "user_id": "ai21",
        "app_id": "complete",
        "model_id": "j2-grande",
        "version_id": "722f731b43bb46daa434702c558576d2"
    },
    AI21_E: {
        "user_id": "ai21",
        "app_id": "complete",
        "model_id": "j2-large",
        "version_id": "22d20fad4d96443cb9f782eec5178f2b"
    },
}

Examples = [
    {
        "title": "Snoop Doog Summary",
        "template": """Rewrite the following paragraph as a rap by Snoop Dogg.
{input}
""",
        "categories": ["Long Form", "Creative"],
    },
]

# This must be within the display() function.
auth = ClarifaiAuthHelper.from_streamlit(st)
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()
lister = ClarifaiResourceLister(stub, auth.user_id, auth.app_id, page_size=16)
st.title("Test out a prompt template")

# with st.form("prompt-form"):
prompt = st.text_area(
    "Enter your prompt template to test out here:",
    placeholder="Here is an example with {input} in the middle. Continue generating ",
    help=
    "You need to place a placeholder {input} in your prompt template. If that is in the middle then two prefix and suffix prompt models will be added to the workflow."
)

model_names = [OPENAI, COHERE, AI21_A, AI21_B, AI21_C, AI21_D, AI21_E]

models = st.multiselect("Select the model(s) you want to use:", model_names)

# button = st.form_submit_button("Create Workflow")


def create_prompt_model(model_id, prompt, position):
  if position not in ["PREFIX", "SUFFIX"]:
    raise Exception("Position must be PREFIX or SUFFIX")

  response = stub.PostModels(
      service_pb2.PostModelsRequest(
          user_app_id=userDataObject,
          models=[
              resources_pb2.Model(
                  id=model_id,
                  model_type_id="prompter",
              ),
          ]))

  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception("PostModels request failed: %r" % response)

  req = service_pb2.PostModelVersionsRequest(
      user_app_id=userDataObject,
      model_id=model_id,
      model_versions=[resources_pb2.ModelVersion(output_info=resources_pb2.OutputInfo())])
  params = json_format.ParseDict({
      "prompt": prompt,
      "position": position,
  }, req.model_versions[0].output_info.params)
  vresponse = stub.PostModelVersions(req)
  if vresponse.status.code != status_code_pb2.SUCCESS:
    raise Exception("PostModelVersions request failed: %r" % vresponse)

  return vresponse.model


def delete_model(model):
  response = stub.DeleteModels(
      service_pb2.DeleteModelsRequest(
          user_app_id=userDataObject,
          ids=[model.id],
      ))
  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception("DeleteModels request failed: %r" % response)


def create_workflow(prefix_model, suffix_model, selected_llm):
  req = service_pb2.PostWorkflowsRequest(
      user_app_id=userDataObject,
      workflows=[
          resources_pb2.Workflow(
              id="test-workflow-" + uuid.uuid4().hex[:10],
              nodes=[
                  resources_pb2.WorkflowNode(
                      id="prefix",
                      model=resources_pb2.Model(
                          id=prefix_model.id,
                          user_id=prefix_model.user_id,
                          app_id=prefix_model.app_id,
                          model_version=resources_pb2.ModelVersion(
                              id=prefix_model.model_version.id,
                              user_id=prefix_model.user_id,
                              app_id=prefix_model.app_id,
                          ),
                      ),
                  ),
                  resources_pb2.WorkflowNode(
                      id="suffix",
                      model=resources_pb2.Model(
                          id=suffix_model.id,
                          user_id=suffix_model.user_id,
                          app_id=suffix_model.app_id,
                          model_version=resources_pb2.ModelVersion(
                              id=suffix_model.model_version.id,
                              user_id=suffix_model.user_id,
                              app_id=suffix_model.app_id,
                          ),
                      ),
                      node_inputs=[resources_pb2.NodeInput(node_id="prefix",)],
                  ),
                  resources_pb2.WorkflowNode(
                      id="llm",
                      model=resources_pb2.Model(
                          id=API_INFO[selected_llm]["model_id"],
                          user_id=API_INFO[selected_llm]["user_id"],
                          app_id=API_INFO[selected_llm]["app_id"],
                          model_version=resources_pb2.ModelVersion(
                              id=API_INFO[selected_llm]["version_id"],
                              user_id=API_INFO[selected_llm]["user_id"],
                              app_id=API_INFO[selected_llm]["app_id"],
                          ),
                      ),
                      node_inputs=[resources_pb2.NodeInput(node_id="suffix",)],
                  ),
              ],
          ),
      ],
  )

  if DEBUG:
    st.json(json_format.MessageToDict(req, preserving_proto_field_name=True))
  response = stub.PostWorkflows(req)
  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception("PostWorkflows request failed: %r" % response)
  if DEBUG:
    st.json(json_format.MessageToDict(response, preserving_proto_field_name=True))

  return response.workflows[0]


def delete_workflow(workflow):
  response = stub.DeleteWorkflows(
      service_pb2.DeleteWorkflowsRequest(
          user_app_id=userDataObject,
          ids=[workflow.id],
      ))
  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception("DeleteWorkflows request failed: %r" % response)


def run_workflow(input_text, workflow):
  response = stub.PostWorkflowResults(
      service_pb2.PostWorkflowResultsRequest(
          user_app_id=userDataObject,
          workflow_id=workflow.id,
          inputs=[
              resources_pb2.Input(
                  data=resources_pb2.Data(text=resources_pb2.Text(raw=input_text,),),),
          ],
      ))
  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception("PostWorkflowResults request failed: %r" % response)

  if DEBUG:
    st.json(json_format.MessageToDict(response, preserving_proto_field_name=True))

  return response


def run_model(input_text, model):
  response = stub.PostModelOutputs(
      service_pb2.PostModelOutputsRequest(
          user_app_id=userDataObject,
          model_id=model.id,
          inputs=[
              resources_pb2.Input(
                  data=resources_pb2.Data(text=resources_pb2.Text(raw=input_text,),),),
          ],
      ))
  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception("PostModelOutputs request failed: %r" % response)

  if DEBUG:
    st.json(json_format.MessageToDict(response, preserving_proto_field_name=True))

  return response


workflows = []
if prompt and models:

  if prompt.find("{input}") == -1:
    st.error("You need to place a placeholder {input} in your prompt template.")
    st.stop()

  if len(models) == 0:
    st.error("You need to select at least one model.")
    st.stop()

  prefix = prompt[:prompt.find("{input}")]
  suffix = prompt[prompt.find("{input}") + len("{input}"):]

  if DEBUG:
    st.write("Prefix:", prefix)
    st.write("Suffix:", suffix)

  prefix_model = create_prompt_model("test-prefix-" + uuid.uuid4().hex, prefix, "PREFIX")
  if DEBUG:
    st.write("PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP")
    st.json(json_format.MessageToDict(prefix_model, preserving_proto_field_name=True))

  suffix_model = create_prompt_model("test-suffix-" + uuid.uuid4().hex, suffix, "SUFFIX")
  if DEBUG:
    st.write("SSSSSSSSSSSSSSSSSSSSSSSSSSSSS")
    st.json(json_format.MessageToDict(suffix_model, preserving_proto_field_name=True))

  for model in models:
    workflows.append(create_workflow(prefix_model, suffix_model, model))

  st.success(
      f"Created {len(workflows)} workflows! Now ready to test it out by inputing some text below")

inp = st.text_input(
    "Try out your new workflow by providing some input:",
    help="This will be used as the input to the {input} placeholder in your prompt template.")

if prompt and models and inp:

  st.header("Completions:")
  completions = []
  for workflow in workflows:
    if DEBUG:
      prefix_prediction = run_model(inp, prefix_model)
      st.write("Prefix:")
      st.json(json_format.MessageToDict(prefix_prediction, preserving_proto_field_name=True))

      suffix_prediction = run_model(inp, suffix_model)
      st.write("Suffix:")
      st.json(json_format.MessageToDict(suffix_prediction, preserving_proto_field_name=True))

    prediction = run_workflow(inp, workflow)
    model_url = f"https://clarifai.com/{workflow.nodes[2].model.user_id}/{workflow.nodes[2].model.app_id}/models/{workflow.nodes[2].model.id}"
    # /versions/{workflow.nodes[2].model.model_version.id}"
    st.write(f"Completion from {model_url}:")
    if DEBUG:
      st.json(json_format.MessageToDict(prediction, preserving_proto_field_name=True))
    completion = prediction.results[0].outputs[2].data.text.raw
    completions.append({"model": model_url, "completion": completion})
    st.info(completion)

  st.dataframe(completions)

  # Cleanup so we don't have tons of junk in this app
  for workflow in workflows:
    delete_workflow(workflow)
  delete_model(prefix_model)
  delete_model(suffix_model)
