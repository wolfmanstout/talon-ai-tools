import base64
import json
import logging
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Literal, NotRequired, Optional, TypedDict

import requests
from talon import actions, app, clip, resource, settings

from ..lib.pureHelpers import strip_markdown
from .modelState import GPTState
from .modelTypes import GPTMessage, GPTMessageItem

""""
All functions in this this file have impure dependencies on either the model or the talon APIs
"""


# TypedDict definition for model configuration
class ModelConfig(TypedDict):
    name: str
    model_id: NotRequired[str]
    system_prompt: NotRequired[str]
    llm_options: NotRequired[dict[str, Any]]
    llm_plugins: NotRequired[list[str]]
    api_options: NotRequired[dict[str, Any]]


@dataclass
class Content:
    text: Optional[str] = None
    image_bytes: Optional[bytes] = None
    fragment: Optional[str] = None
    attachment: Optional[str] = None


@dataclass
class Prompt:
    user_prompt: Optional[str] = None
    content: Optional[Content] = None
    template: Optional[str] = None

    def __post_init__(self):
        """Validate that at least one of user_prompt or template is provided"""
        if self.user_prompt is None and self.template is None:
            raise ValueError("Either user_prompt or template must be provided")


# Path to the models.json file
MODELS_PATH = Path(__file__).parent.parent / "models.json"

# Store loaded model configurations
model_configs: dict[str, ModelConfig] = {}


def load_model_config(f: IO) -> None:
    """
    Load model configurations from models.json
    """
    global model_configs
    try:
        content = f.read()
        configs = json.loads(content)
        # Convert list to dictionary with name as key
        model_configs = {config["name"]: config for config in configs}
    except Exception as e:
        notify(f"Failed to load models.json: {e!r}")
        model_configs = {}


def ensure_models_file_exists():
    if not MODELS_PATH.exists():
        with open(MODELS_PATH, "w") as f:
            f.write("[]")


ensure_models_file_exists()


# Set up file watcher to reload configuration when models.json changes
@resource.watch(str(MODELS_PATH))
def on_update(f: IO):
    load_model_config(f)


def resolve_model_name(model: str) -> str:
    """
    Get the actual model name from the model list value.
    """
    if model == "model":
        # Check for deprecated setting first for backward compatibility
        openai_model: str = settings.get("user.openai_model")  # type: ignore
        if openai_model != "do_not_use":
            logging.warning(
                "The setting 'user.openai_model' is deprecated. Please use 'user.model_default' instead."
            )
            model = openai_model
        else:
            model = settings.get("user.model_default")  # type: ignore
    return model


def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """
    Get the configuration for a specific model from the loaded configs
    """
    return model_configs.get(model_name)


def messages_to_string(messages: list[GPTMessageItem]) -> str:
    """Format messages as a string"""
    formatted_messages = []
    for message in messages:
        if message.get("type") == "image_url":
            formatted_messages.append("image")
        else:
            formatted_messages.append(message.get("text", ""))
    return "\n\n".join(formatted_messages)


def notify(message: str):
    """Send a notification to the user. Defaults the Andreas' notification system if you have it installed"""
    try:
        actions.user.notify(message)
    except Exception:
        app.notify(message)
    # Log in case notifications are disabled
    print(message)


def get_token() -> str:
    """Get the OpenAI API key from the environment"""
    try:
        return os.environ["OPENAI_API_KEY"]
    except KeyError:
        message = "GPT Failure: env var OPENAI_API_KEY is not set."
        notify(message)
        raise Exception(message)


def format_messages(
    role: Literal["user", "system", "assistant"], messages: list[GPTMessageItem]
) -> GPTMessage:
    return {
        "role": role,
        "content": messages,
    }


def format_message(content: str) -> GPTMessageItem:
    return {"type": "text", "text": content}


def extract_message(content: GPTMessageItem) -> str:
    return content.get("text", "")


@dataclass
class ClipboardContent:
    text: Optional[str] = None
    html: Optional[str] = None
    image_bytes: Optional[bytes] = None


@dataclass
class InlineContent:
    text: Optional[str] = None
    image_bytes: Optional[bytes] = None

    def has_content(self) -> bool:
        return any([self.text, self.image_bytes])


def fetch_from_clipboard() -> ClipboardContent:
    """Get all available content from clipboard at once"""
    content = ClipboardContent()

    # Get image if available
    clipped_image = clip.image()
    if clipped_image:
        content.image_bytes = clipped_image.encode().data()

    # Get text if available
    if clip.text():
        content.text = clip.text()

    # Get HTML if available
    mime = clip.mime()
    if mime and mime.html:
        content.html = extract_clip_html(mime.html)

    return content


def fetch_from_selection() -> ClipboardContent:
    """Get all available content from selection"""
    content = ClipboardContent()

    # Capture clipboard with selection
    timeout = settings.get("user.selected_text_timeout")
    with clip.capture(timeout) as s:
        actions.edit.copy()

    # Get text
    try:
        content.text = s.text()
    except clip.NoChange:
        pass

    # Get HTML if available
    try:
        if s.mime() and s.mime().html:
            content.html = extract_clip_html(s.mime().html)
    except clip.NoChange:
        pass

    return content


def convert_content(
    content: ClipboardContent, format_type: Optional[str]
) -> InlineContent:
    """Convert ClipboardContent to InlineContent with desired format"""
    # Handle specific format requests
    if format_type == "html":
        if content.html:
            return InlineContent(text=content.html)
        error_msg = "No HTML content found"
        notify(error_msg)
        raise Exception(error_msg)

    elif format_type == "markdown":
        if content.html:
            markdown = convert_html_to_markdown(content.html)
            if markdown:
                return InlineContent(text=markdown)
            error_msg = "Failed to convert HTML to markdown"
            notify(error_msg)
            raise Exception(error_msg)
        error_msg = "No HTML content found to convert to markdown"
        notify(error_msg)
        raise Exception(error_msg)

    elif format_type == "text":
        if content.text:
            return InlineContent(text=content.text)
        error_msg = "No text content found"
        notify(error_msg)
        raise Exception(error_msg)

    # For unspecified format type
    elif format_type is None:
        # Prioritize image if available
        if content.image_bytes:
            return InlineContent(image_bytes=content.image_bytes)

        # Try auto-converting HTML to markdown
        if content.html:
            markdown = convert_html_to_markdown(content.html)
            if markdown:
                return InlineContent(text=markdown)
            # If conversion fails, just log and continue to text
            warning = "Failed to convert HTML to markdown, falling back to plain text"
            notify(warning)
            logging.warning(warning)

        # Use text as fallback
        if content.text:
            return InlineContent(text=content.text)

        return InlineContent()

    else:
        error_msg = f"Unknown format type: {format_type}"
        notify(error_msg)
        raise Exception(error_msg)


def extract_clip_html(html_data: str) -> str:
    """Extract HTML content from clipboard, handling Windows metadata"""
    if not html_data:
        return ""

    # Windows uses a special HTML clipboard format with headers and offsets
    if platform.system() == "Windows":
        # Parse the header to find StartFragment and EndFragment offsets
        lines = html_data.split("\r\n")
        start_fragment = None
        end_fragment = None

        for line in lines:
            if line.startswith("StartFragment:"):
                start_fragment = int(line.split(":")[1])
            elif line.startswith("EndFragment:"):
                end_fragment = int(line.split(":")[1])

        if start_fragment is None or end_fragment is None:
            return ""

        # Extract the HTML fragment using the offsets
        html_fragment = html_data[start_fragment:end_fragment]
        return html_fragment
    else:
        # On Mac and other platforms, HTML mime data is the raw HTML
        return html_data


def convert_html_to_markdown(html: str) -> Optional[str]:
    """Convert HTML to markdown using markdownify CLI"""
    # Configure output encoding
    process_env = os.environ.copy()
    if platform.system() == "Windows":
        process_env["PYTHONUTF8"] = "1"  # For Python 3.7+ to enable UTF-8 mode
    # On other platforms, UTF-8 is also the common/expected encoding.
    text_encoding = "utf-8"

    try:
        markdownify_path: str = settings.get("user.model_markdownify_path")  # type: ignore
        markdown = subprocess.check_output(
            [markdownify_path],
            input=html,
            encoding=text_encoding,
            stderr=subprocess.PIPE,
            creationflags=(
                subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0  # type: ignore
            ),
            env=process_env if platform.system() == "Windows" else None,
        ).strip()
        return markdown
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else str(e)
        logging.error(f"Error converting HTML to markdown: {error_msg}")
        return None
    except Exception as e:
        logging.error(f"Error converting HTML to markdown: {str(e)}")
        return None


def send_request(
    prompt: Prompt,
    model: str,
    thread: str,
    destination: str = "",
) -> GPTMessageItem:
    """Generate run a GPT request and return the response"""
    model = resolve_model_name(model)

    continue_thread = thread == "continueLast"

    notification = "GPT Task Started"
    if len(GPTState.context) > 0:
        notification += ": Reusing Stored Context"

    # Use specified model if provided
    if model:
        notification += f", Using model: {model}"

    if settings.get("user.model_verbose_notifications"):
        notify(notification)

    # Get model configuration if available
    config = get_model_config(model)

    language = actions.code.language()
    language_context = (
        f"The user is currently in a code editor for the programming language: {language}."
        if language != ""
        else None
    )
    application_context = f"The following describes the currently focused application:\n\n{actions.user.talon_get_active_context()}"
    snippet_context = (
        "\n\nPlease return the response as a snippet with placeholders. A snippet can control cursors and text insertion using constructs like tabstops ($1, $2, etc., with $0 as the final position). Linked tabstops update together. Placeholders, such as ${1:foo}, allow easy changes and can be nested (${1:another ${2:}}). Choices, using ${1|one,two,three|}, prompt user selection."
        if destination == "snip"
        else None
    )

    system_message = "\n\n".join(
        [
            item
            for item in [
                (
                    config["system_prompt"]
                    if config and "system_prompt" in config
                    else settings.get("user.model_system_prompt")
                ),
                language_context,
                application_context,
                snippet_context,
            ]
            + actions.user.gpt_additional_user_context()
            + [context.get("text") for context in GPTState.context]
            if item
        ]
    )

    model_endpoint: str = settings.get("user.model_endpoint")  # type: ignore
    if model_endpoint == "llm":
        response = send_request_to_llm_cli(
            prompt, system_message, model, continue_thread
        )
    else:
        if continue_thread:
            notify(
                "Warning: Thread continuation is only supported when using setting user.model_endpoint = 'llm'"
            )
        response = send_request_to_api(prompt, system_message, model)

    return response


def send_request_to_api(
    prompt: Prompt,
    system_message: str,
    model: str,
) -> GPTMessageItem:
    """Send a request to the model API endpoint and return the response"""
    # Get model configuration if available
    config = get_model_config(model)

    # Use model_id from configuration if available
    model_id = config["model_id"] if config and "model_id" in config else model

    # Create GPTMessageItem from user prompt
    if prompt.user_prompt:
        gpt_prompt = format_message(prompt.user_prompt)
    else:
        notify("GPT Failure: No user prompt provided.")
        raise ValueError("No prompt provided.")

    # Prepare content for API request
    api_content: list[GPTMessageItem] = [gpt_prompt]
    if prompt.content:
        if prompt.content.image_bytes:
            # If we are processing an image, add it as a second message
            base64_image = base64.b64encode(prompt.content.image_bytes).decode("utf-8")
            api_content = [
                gpt_prompt,
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/;base64,{base64_image}"},
                },
            ]
        elif prompt.content.text:
            # If we are processing text content, add it to the same message
            gpt_prompt["text"] = (
                gpt_prompt["text"] + '\n\n"""' + prompt.content.text + '"""'  # type: ignore a Prompt has to be of type text
            )
            api_content = [gpt_prompt]
        else:
            notify("GPT Failure: Check the Talon Log. Invalid content provided.")
            raise ValueError(
                f"Invalid content type. Only text and image content are supported. Received: {prompt.content}"
            )

    # Create request
    request = GPTMessage(
        role="user",
        content=api_content,
    )

    data = {
        "messages": (
            [
                format_messages(
                    "system",
                    [GPTMessageItem(type="text", text=system_message)],
                ),
            ]
            if system_message
            else []
        )
        + [request],
        "max_tokens": 2024,
        "n": 1,
        "model": model_id,
    }

    # Check for deprecated temperature setting
    temperature: float = settings.get("user.model_temperature")  # type: ignore
    if temperature != -1.0:
        logging.warning(
            "The setting 'user.model_temperature' is deprecated. Please configure temperature in models.json instead."
        )
        data["temperature"] = temperature

    # Apply API options from configuration if available
    if config and "api_options" in config:
        data.update(config["api_options"])

    if GPTState.debug_enabled:
        print(data)

    url: str = settings.get("user.model_endpoint")  # type: ignore
    headers = {"Content-Type": "application/json"}
    token = get_token()
    # If the model endpoint is Azure, we need to use a different header
    if "azure.com" in url:
        headers["api-key"] = token
    else:
        headers["Authorization"] = f"Bearer {token}"

    raw_response = requests.post(url, headers=headers, data=json.dumps(data))

    match raw_response.status_code:
        case 200:
            if settings.get("user.model_verbose_notifications"):
                notify("GPT Task Completed")
            resp = raw_response.json()["choices"][0]["message"]["content"].strip()
            formatted_resp = strip_markdown(resp)
            return format_message(formatted_resp)
        case _:
            notify("GPT Failure: Check the Talon Log")
            raise Exception(raw_response.json())


def send_request_to_llm_cli(
    prompt: Prompt,
    system_message: str,
    model: str,
    continue_thread: bool,
) -> GPTMessageItem:
    """Send a request to the LLM CLI tool and return the response"""
    # Get model configuration if available
    config = get_model_config(model)

    # Use model_id from configuration if available
    model_id = config["model_id"] if config and "model_id" in config else model

    # Build command
    command: list[str] = [settings.get("user.model_llm_path")]  # type: ignore
    if continue_thread:
        command.append("-c")

    # Add model option
    command.extend(["-m", model_id])

    # Check for deprecated temperature setting
    temperature: float = settings.get("user.model_temperature")  # type: ignore
    if temperature != -1.0:
        logging.warning(
            "The setting 'user.model_temperature' is deprecated. Please configure temperature in models.json instead."
        )
        command.extend(["-o", "temperature", str(temperature)])

    # Apply llm_options from configuration if available
    if config and "llm_options" in config:
        for key, value in config["llm_options"].items():
            if isinstance(value, bool):
                if value:
                    command.extend(["-o", key, "true"])
                else:
                    command.extend(["-o", key, "false"])
            else:
                command.extend(["-o", key, str(value)])

    # Handle template if specified
    if prompt.template:
        command.extend(["-t", prompt.template])

    # Add the prompt
    cmd_input: bytes | None = None
    if prompt.content:
        if prompt.content.fragment:
            # For direct fragment use, pass the fragment name directly
            command.extend(["-f", prompt.content.fragment])
        elif prompt.content.image_bytes:
            # For image content
            command.extend(["-a", "-"])
            cmd_input = prompt.content.image_bytes
        elif prompt.content.attachment:
            # For external attachment URL
            command.extend(["-a", prompt.content.attachment])
        elif prompt.content.text and prompt.user_prompt:
            # For regular text content with a user prompt, combine them
            user_prompt_text = (
                prompt.user_prompt + '\n\n"""' + prompt.content.text + '"""'
            )
            command.append(user_prompt_text)
        elif prompt.content.text:
            # Just content text, no user prompt
            command.append(prompt.content.text)
        else:
            notify("GPT Failure: Check the Talon Log. Invalid content provided.")
            raise ValueError(f"Invalid content type. Received: {prompt.content}")
    # If we have a user prompt but no content
    elif prompt.user_prompt:
        command.append(prompt.user_prompt)

    # Add system message if available
    if system_message and not prompt.template:
        command.extend(["-s", system_message])

    if GPTState.debug_enabled:
        print(command)

    # Configure output encoding
    process_env = os.environ.copy()
    if platform.system() == "Windows":
        process_env["PYTHONUTF8"] = "1"  # For Python 3.7+ to enable UTF-8 mode
    # On other platforms, UTF-8 is also the common/expected encoding.
    output_encoding = "utf-8"

    # Handle LLM plugins configuration
    base_plugins: str = settings.get("user.model_llm_plugins")  # type: ignore
    model_plugins = config.get("llm_plugins") if config else None
    env_modified = False

    if base_plugins != "<all>":
        plugins_list = [p.strip() for p in base_plugins.split(",") if p.strip()]
        if model_plugins:
            plugins_list.extend(model_plugins)

        # Set environment variable even if plugins_list is empty (for empty string case)
        process_env["LLM_LOAD_PLUGINS"] = ",".join(plugins_list)
        env_modified = True

    # Execute command and capture output.
    try:
        resp = subprocess.check_output(
            command,
            input=cmd_input,
            encoding=output_encoding,
            stderr=subprocess.PIPE,
            creationflags=(
                subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0  # type: ignore
            ),
            env=(
                process_env
                if (platform.system() == "Windows" or env_modified)
                else None
            ),
        ).strip()
        if settings.get("user.model_verbose_notifications"):
            notify("GPT Task Completed")
        formatted_resp = strip_markdown(resp)
        return format_message(formatted_resp)
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else str(e)
        notify(f"GPT Failure: {error_msg}")
        raise e
    except Exception as e:
        notify("GPT Failure: Check the Talon Log")
        raise e
