import os
from typing import Any, Optional

from talon import Module, actions, clip, settings

from ..lib.HTMLBuilder import Builder
from ..lib.modelConfirmationGUI import confirmation_gui
from ..lib.modelHelpers import (
    Content,
    InlineContent,
    Prompt,
    convert_content,
    extract_message,
    fetch_from_clipboard,
    fetch_from_selection,
    format_message,
    messages_to_string,
    notify,
    send_request,
)
from ..lib.modelState import GPTState
from ..lib.modelTypes import GPTMessageItem
from ..lib.talonSettings import ContentSpec


def resolve_source(source: str, format_type: Optional[str] = None) -> InlineContent:
    """Resolve content from a source identifier with optional format conversion"""
    match source:
        case "clipboard":
            content = convert_content(fetch_from_clipboard(), format_type)
            if not content.has_content():
                error_msg = "GPT Failure: Clipboard is empty"
                notify(error_msg)
                raise Exception(
                    f"{error_msg}. User applied a prompt to the clipboard, but it was empty"
                )
            return content

        case "context":
            if GPTState.context == []:
                error_msg = "GPT Failure: Context is empty"
                notify(error_msg)
                raise Exception(
                    f"{error_msg}. User applied a prompt to the phrase context, but there was no context stored"
                )
            return InlineContent(text=messages_to_string(GPTState.context))

        case "gptResponse":
            if GPTState.last_response == "":
                error_msg = "GPT Failure: No GPT response stored"
                notify(error_msg)
                raise Exception(
                    f"{error_msg}. User applied a prompt to the phrase GPT response, but there was no GPT response stored"
                )
            return InlineContent(text=GPTState.last_response)

        case "lastTalonDictation":
            last_output = actions.user.get_last_phrase()
            if last_output:
                actions.user.clear_last_phrase()
                return InlineContent(text=last_output)
            else:
                error_msg = "GPT Failure: No last dictation to reformat"
                notify(error_msg)
                raise Exception(
                    f"{error_msg}. User applied a prompt to the phrase last Talon Dictation, but there was no text to reformat"
                )

        case "this" | _:
            # This is the default source, so we allow empty content to be returned.
            return convert_content(fetch_from_selection(), format_type)


mod = Module()
mod.tag(
    "model_window_open",
    desc="Tag for enabling the model window commands when the window is open",
)


def gpt_query(
    prompt: Prompt,
    model: str,
    thread: str,
    destination: str = "",
):
    """Send a prompt to the GPT API and return the response"""

    # Reset state before pasting
    GPTState.last_was_pasted = False

    response = send_request(prompt, model, thread, destination)
    GPTState.last_response = extract_message(response)
    return response


@mod.action_class
class UserActions:
    def gpt_generate_shell(text_to_process: str, model: str, thread: str) -> str:
        """Generate a shell command from a spoken instruction"""
        shell_name = settings.get("user.model_shell_default")
        if shell_name is None:
            raise Exception("GPT Error: Shell name is not set. Set it in the settings.")

        prompt_text = f"""
        Generate a {shell_name} shell command that will perform the given task.
        Only include the code. Do not include any comments, backticks, or natural language explanations. Do not output the shell name, only the code that is valid {shell_name}.
        Condense the code into a single line such that it can be ran in the terminal.
        """

        prompt = Prompt(user_prompt=prompt_text, content=Content(text=text_to_process))

        result = gpt_query(
            prompt,
            model,
            thread,
        )
        return extract_message(result)

    def gpt_generate_sql(text_to_process: str, model: str, thread: str) -> str:
        """Generate a SQL query from a spoken instruction"""

        prompt_text = """
       Generate SQL to complete a given request.
       Output only the SQL in one line without newlines.
       Do not output comments, backticks, or natural language explanations.
       Prioritize SQL queries that are database agnostic.
        """

        prompt = Prompt(user_prompt=prompt_text, content=Content(text=text_to_process))

        return gpt_query(
            prompt,
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
        prompt_text: str,
        model: str,
        thread: str,
        source: Optional[ContentSpec] | str = None,
        destination: str = "",
        template: Optional[str] = None,
    ) -> GPTMessageItem:
        """Apply a prompt or template to arbitrary text

        The prompt_text is required but can be empty if a template is provided.

        If source is a non-empty string, it will be treated as direct text content.
        If source is a ContentSpec, it will be processed through gpt_get_source_text.
        """
        # Validate that we have a prompt_text or template
        if not prompt_text and not template:
            error = "Either prompt_text or template must be provided"
            notify(error)
            raise ValueError(error)

        if source:
            if isinstance(source, str) and source:
                content = Content(text=source)
            else:
                content = actions.user.gpt_get_source_text(source)
                if not content:
                    error = f"No content for source: {source}"
                    notify(error)
                    raise Exception(error)
        else:
            content = actions.user.gpt_get_source_text(ContentSpec(source="this"))

        prompt = Prompt(user_prompt=prompt_text, content=content, template=template)

        response = gpt_query(prompt, model, thread, destination)

        actions.user.gpt_insert_response(response, destination)
        return response

    def gpt_apply_prompt_for_cursorless(
        prompt_text: str,
        model: str,
        thread: str,
        source: list[str],
        template: Optional[str] = None,
    ) -> str:
        """Apply a prompt or template to text from Cursorless and return a string result.
        This function is specifically designed for Cursorless integration
        and does not trigger insertion actions.

        The prompt_text is required but can be empty if a template is provided.
        """
        # Validate that we have a prompt_text or template
        if not prompt_text and not template:
            error = "Either prompt_text or template must be provided"
            notify(error)
            raise ValueError(error)

        # Join the list into a single string
        source_text = "\n".join(source)

        # Create content with the text
        content = Content(text=source_text)

        # Create the prompt object
        prompt = Prompt(user_prompt=prompt_text, content=content, template=template)

        # Send the request but don't insert the response (Cursorless will handle insertion)
        response = gpt_query(prompt, model, thread, "")

        # Return just the text string
        return extract_message(response)

    def gpt_pass(source: str = "", destination: str = "") -> None:
        """Passes a response from source to destination"""
        content_spec = ContentSpec(source=source)
        content = actions.user.gpt_get_source_text(content_spec)
        if content and content.text:
            actions.user.gpt_insert_response(format_message(content.text), destination)
        else:
            error = f"No text found for source: {source}"
            notify(error)
            raise Exception(error)

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
        prompt_text = f"""The last phrase was written using voice dictation. It has an error with spelling, grammar, or just general misrecognition due to a lack of context. Please reformat the following text to correct the error with the context that it was {how_to_reformat}."""
        last_output = actions.user.get_last_phrase()
        if last_output:
            actions.user.clear_last_phrase()

            prompt = Prompt(user_prompt=prompt_text, content=Content(text=last_output))

            return extract_message(
                gpt_query(
                    prompt,
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
                builder.model_result(message_text_no_images)
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
        content_spec: ContentSpec,
    ) -> Optional[Content]:
        """Get the source content based on the ContentSpec provided by the modelSource capture"""
        # If it's a direct fragment, just return the fragment name
        if content_spec.fragment:
            return Content(fragment=content_spec.fragment)

        # If it's a source as fragment setup
        elif content_spec.source_as_fragment:
            # Get the content for the source
            source = content_spec.source_as_fragment.source
            inline_content = resolve_source(source)

            # Can't use images as fragments with prefixes
            if inline_content.image_bytes:
                notify("GPT Failure: Can't use images as fragments with prefixes")
                raise Exception("Images cannot be used as fragments with prefixes")

            # Only continue if we have text content
            if not inline_content.text:
                return None

            # Create the fragment by combining the prefix with the content
            # Strip whitespace from the text to avoid issues with extra spaces
            fragment = (
                f"{content_spec.source_as_fragment.prefix}{inline_content.text.strip()}"
            )
            return Content(fragment=fragment)

        # If it's specified as an attachment
        elif content_spec.attachment:
            # Get the source content
            inline_content = resolve_source(content_spec.attachment)

            # For images, return an error - attachments should be URLs or files
            if inline_content.image_bytes:
                notify(
                    "GPT Failure: Can't use images directly as attachments, use URLs or file paths instead"
                )
                return None

            # For text content, pass it directly as attachment
            elif inline_content.text:
                return Content(attachment=inline_content.text.strip())
            else:
                return None

        # If it's a formatted source
        elif content_spec.formatted_source:
            source = content_spec.formatted_source.source
            format_type = content_spec.formatted_source.format

            # Get the content with the specified format
            inline_content = resolve_source(source, format_type)

            if inline_content.image_bytes:
                return Content(image_bytes=inline_content.image_bytes)
            elif inline_content.text:
                return Content(text=inline_content.text)
            else:
                return None

        # Regular source resolution
        elif content_spec.source:
            inline_content = resolve_source(content_spec.source)

            if inline_content.image_bytes:
                return Content(image_bytes=inline_content.image_bytes)
            elif inline_content.text:
                return Content(text=inline_content.text)
            else:
                return None

        else:
            error_msg = "Invalid ContentSpec: must specify either fragment, source, source_as_fragment, attachment, or formatted_source"
            notify(error_msg)
            raise ValueError(error_msg)
