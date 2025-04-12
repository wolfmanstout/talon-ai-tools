import os
from typing import Any, Literal, NotRequired, Optional, TypedDict

from talon import Module, actions, clip, settings

from ..lib.HTMLBuilder import Builder
from ..lib.modelConfirmationGUI import confirmation_gui
from ..lib.modelHelpers import (
    SourceInfo,
    extract_message,
    format_clipboard,
    format_message,
    messages_to_string,
    notify,
    send_request,
)
from ..lib.modelState import GPTState
from ..lib.modelTypes import GPTMessageItem


# TypedDict for the source descriptor returned from the modelSource capture
class SourceDescriptor(TypedDict):
    type: Literal["source", "fragment", "source_as_fragment"]
    value: str
    prefix: NotRequired[str]  # Required when type is "source_as_fragment"


# Helper function to create a source info of type "source" with text content
def create_source_info_text(text: str) -> SourceInfo:
    """Create a source info with 'source' type and text content"""
    return {"type": "source", "content": format_message(text)}


mod = Module()
mod.tag(
    "model_window_open",
    desc="Tag for enabling the model window commands when the window is open",
)


def gpt_query(
    prompt: GPTMessageItem,
    source_info: Optional[SourceInfo],
    model: str,
    thread: str,
    destination: str = "",
):
    """Send a prompt to the GPT API and return the response"""

    # Reset state before pasting
    GPTState.last_was_pasted = False

    response = send_request(prompt, source_info, model, thread, destination)
    GPTState.last_response = extract_message(response)
    return response


@mod.action_class
class UserActions:
    def gpt_generate_shell(text_to_process: str, model: str, thread: str) -> str:
        """Generate a shell command from a spoken instruction"""
        shell_name = settings.get("user.model_shell_default")
        if shell_name is None:
            raise Exception("GPT Error: Shell name is not set. Set it in the settings.")

        prompt = f"""
        Generate a {shell_name} shell command that will perform the given task.
        Only include the code. Do not include any comments, backticks, or natural language explanations. Do not output the shell name, only the code that is valid {shell_name}.
        Condense the code into a single line such that it can be ran in the terminal.
        """

        result = gpt_query(
            format_message(prompt),
            create_source_info_text(text_to_process),
            model,
            thread,
        )
        return extract_message(result)

    def gpt_generate_sql(text_to_process: str, model: str, thread: str) -> str:
        """Generate a SQL query from a spoken instruction"""

        prompt = """
       Generate SQL to complete a given request.
       Output only the SQL in one line without newlines.
       Do not output comments, backticks, or natural language explanations.
       Prioritize SQL queries that are database agnostic.
        """

        return gpt_query(
            format_message(prompt),
            create_source_info_text(text_to_process),
            model,
            thread,
        ).get("text", "")

    def gpt_start_debug():
        """Enable debug logging"""
        GPTState.start_debug()

    def gpt_stop_debug():
        """Disable debug logging"""
        GPTState.stop_debug()

    def gpt_clear_context():
        """Reset the stored context"""
        GPTState.clear_context()

    def gpt_push_context(context: str | list[str]):
        """Add the selected text to the stored context"""
        if isinstance(context, list):
            context = "\n".join(context)
        GPTState.push_context(format_message(context))

    def gpt_additional_user_context() -> list[str]:
        """This is an override function that can be used to add additional context to the prompt"""
        return []

    def gpt_select_last() -> None:
        """select all the text in the last GPT output"""
        if not GPTState.last_was_pasted:
            notify("Tried to select GPT output, but it was not pasted in an editor")
            return

        lines = GPTState.last_response.split("\n")
        for _ in lines[:-1]:
            actions.edit.extend_up()
        actions.edit.extend_line_end()
        for _ in lines[0]:
            actions.edit.extend_left()

    def gpt_apply_prompt(
        prompt: str,
        model: str,
        thread: str,
        source: Optional[SourceDescriptor] | str = None,  # Allow empty string
        destination: str = "",
    ) -> GPTMessageItem:
        """Apply an arbitrary prompt to arbitrary text"""

        source_info = actions.user.gpt_get_source_text(source or None)

        response = gpt_query(
            format_message(prompt), source_info, model, thread, destination
        )

        actions.user.gpt_insert_response(response, destination)
        return response

    def gpt_apply_prompt_for_cursorless(
        prompt: str,
        model: str,
        thread: str,
        source: list[str],
    ) -> str:
        """Apply a prompt to text from Cursorless and return a string result.
        This function is specifically designed for Cursorless integration
        and does not trigger insertion actions."""

        # Join the list into a single string
        source_text = "\n".join(source)

        # Create a simple source_info with direct source
        source_info = create_source_info_text(source_text)

        # Send the request but don't insert the response (Cursorless will handle insertion)
        response = gpt_query(format_message(prompt), source_info, model, thread, "")

        # Return just the text string
        return extract_message(response)

    def gpt_pass(source: str = "", destination: str = "") -> None:
        """Passes a response from source to destination"""
        source_descriptor: SourceDescriptor = {"type": "source", "value": source}
        source_info = actions.user.gpt_get_source_text(source_descriptor)
        actions.user.gpt_insert_response(source_info["content"], destination)

    def gpt_help() -> None:
        """Open the GPT help file in the web browser"""
        # get the text from the file and open it in the web browser
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, "lists", "staticPrompt.talon-list")
        with open(file_path, "r") as f:
            lines = f.readlines()[2:]

        builder = Builder()
        builder.h1("Talon GPT Prompt List")
        for line in lines:
            if "##" in line:
                builder.h2(line)
            else:
                builder.p(line)

        builder.render()

    def gpt_reformat_last(how_to_reformat: str, model: str, thread: str) -> str:
        """Reformat the last model output"""
        PROMPT = f"""The last phrase was written using voice dictation. It has an error with spelling, grammar, or just general misrecognition due to a lack of context. Please reformat the following text to correct the error with the context that it was {how_to_reformat}."""
        last_output = actions.user.get_last_phrase()
        if last_output:
            actions.user.clear_last_phrase()
            return extract_message(
                gpt_query(
                    format_message(PROMPT),
                    create_source_info_text(last_output),
                    model,
                    thread,
                )
            )
        else:
            notify("No text to reformat")
            raise Exception("No text to reformat")

    def gpt_insert_response(
        gpt_message: GPTMessageItem,
        method: str = "",
        cursorless_destination: Any = None,
    ) -> None:
        """Insert a GPT result in a specified way"""
        # Use a custom default if nothing is provided and the user has set
        # a different default destination
        if method == "":
            method = settings.get("user.model_default_destination")

        if gpt_message.get("type") != "text":
            actions.app.notify(
                f"Tried to insert an image to {method}, but that is not currently supported. To insert an image to this destination use a prompt to convert it to text."
            )
            return

        message_text_no_images = extract_message(gpt_message)
        match method:
            case "above":
                actions.key("left")
                actions.edit.line_insert_up()
                GPTState.last_was_pasted = True
                actions.user.paste(message_text_no_images)
            case "below":
                actions.key("right")
                actions.edit.line_insert_down()
                GPTState.last_was_pasted = True
                actions.user.paste(message_text_no_images)
            case "clipboard":
                clip.set_text(message_text_no_images)
            case "snip":
                actions.user.insert_snippet(message_text_no_images)
            case "context":
                GPTState.push_context(gpt_message)
            case "newContext":
                GPTState.clear_context()
                GPTState.push_context(gpt_message)
            case "appendClipboard":
                if clip.text() is not None:
                    clip.set_text(clip.text() + "\n" + message_text_no_images)  # type: ignore Unclear why this is throwing a type error in pylance
                else:
                    clip.set_text(message_text_no_images)
            case "browser":
                builder = Builder()
                builder.h1("Talon GPT Result")
                for line in message_text_no_images.split("\n"):
                    builder.p(line)
                builder.render()
            case "textToSpeech":
                try:
                    actions.user.tts(message_text_no_images)
                except KeyError:
                    notify("GPT Failure: text to speech is not installed")

            # Although we can insert to a cursorless destination, the cursorless_target capture
            # Greatly increases DFA compliation times and should be avoided if possible
            case "cursorless":
                actions.user.cursorless_insert(
                    cursorless_destination, message_text_no_images
                )
            # Don't add to the window twice if the thread is enabled
            case "window":
                # If there was prior text in the confirmation GUI and the user
                # explicitly passed new text to the gui, clear the old result
                GPTState.text_to_confirm = message_text_no_images
                actions.user.confirmation_gui_append(message_text_no_images)
            case "chain":
                GPTState.last_was_pasted = True
                actions.user.paste(message_text_no_images)
                actions.user.gpt_select_last()

            case "paste":
                GPTState.last_was_pasted = True
                actions.user.paste(message_text_no_images)
            case "diff":
                GPTState.last_was_pasted = True
                actions.user.draft_editor_open_diff(new=message_text_no_images)
            case "draft":
                GPTState.last_was_pasted = True
                actions.user.draft_editor_open(message_text_no_images)

            # If the user doesn't specify a method assume they want to paste.
            # However if they didn't specify a method when the confirmation gui
            # is showing, assume they don't want anything to be inserted
            case _ if not confirmation_gui.showing:
                GPTState.last_was_pasted = True
                actions.user.paste(message_text_no_images)
            # Don't do anything if none of the previous conditions were valid
            case _:
                pass

    def gpt_get_source_text(
        source_descriptor: Optional[SourceDescriptor],
    ) -> Optional[SourceInfo]:
        """Get the source content based on the source descriptor provided by the modelSource capture"""
        # If source_descriptor is None, return None
        if source_descriptor is None:
            return None

        result_info: SourceInfo = {"type": source_descriptor["type"]}  # type: ignore

        # If it's a direct fragment, just return the fragment name
        if source_descriptor["type"] == "fragment":
            result_info["content"] = format_message(source_descriptor["value"])
            return result_info

        # Get the actual content based on the source identifier
        source_identifier = source_descriptor["value"]
        content = None

        match source_identifier:
            case "clipboard":
                content = format_clipboard()
            case "context":
                if GPTState.context == []:
                    notify("GPT Failure: Context is empty")
                    raise Exception(
                        "GPT Failure: User applied a prompt to the phrase context, but there was no context stored"
                    )
                content = format_message(messages_to_string(GPTState.context))
            case "gptResponse":
                if GPTState.last_response == "":
                    raise Exception(
                        "GPT Failure: User applied a prompt to the phrase GPT response, but there was no GPT response stored"
                    )
                content = format_message(GPTState.last_response)
            case "lastTalonDictation":
                last_output = actions.user.get_last_phrase()
                if last_output:
                    actions.user.clear_last_phrase()
                    content = format_message(last_output)
                else:
                    notify("GPT Failure: No last dictation to reformat")
                    raise Exception(
                        "GPT Failure: User applied a prompt to the phrase last Talon Dictation, but there was no text to reformat"
                    )
            case "this" | _:
                content = format_message(actions.edit.selected_text())

        # Check if content is empty
        if content is None or not content.get("text") and not content.get("image_url"):
            return None

        result_info["content"] = content

        # If it's source as fragment, add the prefix
        if source_descriptor["type"] == "source_as_fragment":
            if "prefix" not in source_descriptor:
                raise ValueError("source_as_fragment must have 'prefix' field")
            result_info["prefix"] = source_descriptor["prefix"]

        return result_info
