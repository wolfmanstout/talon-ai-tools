mode: command
mode: user.dictation_command
-

# Shows the list of available prompts
{user.model} help$: user.gpt_help()

# # Runs a model prompt on the selected text; inserts with paste by default
# #   Example: `model fix grammar below` -> Fixes the grammar of the selected text and pastes below
# #   Example: `model explain this` -> Explains the selected text and pastes in place
# #   Example: `model fix grammar clip to browser` -> Fixes the grammar of the text on the clipboard and opens in browser`
# {user.model} <user.modelPrompt> [{user.modelSource}] [{user.modelDestination}]$:
#     user.gpt_apply_prompt(modelPrompt, modelSource or "", modelDestination or "")

{user.model} [<user.continueThread>] {user.modelAction} <user.modelSimplePrompt> [{user.modelSource}]$:
    user.gpt_apply_prompt(modelSimplePrompt, modelSource or "", modelAction, model, continueThread or "")

# Alternative ordering (for consistency with later variants).
{user.model} [<user.continueThread>] {user.modelAction} with {user.modelSource} <user.modelSimplePrompt>$:
    user.gpt_apply_prompt(modelSimplePrompt, modelSource, modelAction, model, continueThread or "")

# Perform arbitrary prompt. If text is selected, it will be provided.
{user.model} [<user.continueThread>] {user.modelAction} <user.prose>$:
    user.gpt_apply_prompt(prose, "", modelAction, model, continueThread or "")

# Perform arbitrary prompt on something other than the selected text.
{user.model} [<user.continueThread>] {user.modelAction} with {user.modelSource} <user.prose>$:
    user.gpt_apply_prompt(prose, modelSource, modelAction, model, continueThread or "")

# Select the last GPT response so you can edit it further
{user.model} take response: user.gpt_select_last()

# Applies an arbitrary prompt from the clipboard to selected text and pastes the result.
# Useful for applying complex/custom prompts that need to be drafted in a text editor.
{user.model} apply [from] clip$:
    prompt = clip.text()
    text = edit.selected_text()
    result = user.gpt_apply_prompt(prompt, text)
    user.paste(result)

# Reformat the last dictation with additional context or formatting instructions
{user.model} [nope] that was <user.text>$:
    result = user.gpt_reformat_last(text)
    user.paste(result)

# Enable debug logging so you can more details about messages being sent
{user.model} start debug: user.gpt_start_debug()

# Disable debug logging
{user.model} stop debug: user.gpt_stop_debug()
